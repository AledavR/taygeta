"""Preparación, rasterización y geometría de figuras (sección 4.1).

Responsabilidades:

* Abrir documentos PDF (y también imágenes PNG/JPG) con PyMuPDF (``fitz``).
* Rasterizar las páginas a 300 DPI por defecto.
* Desnormalizar las BBox de escala 0-1000 al tamaño real de la página aplicando
  un padding preventivo del 1.5 %.
* Mantener el índice físico real de página basado en cero y conservar las
  coordenadas necesarias para asociar cada figura con el enunciado exacto.
* Detectar figuras y diagramas relevantes y asignar placeholders ``{{fig:N}}``
  en orden de lectura.
* Crear recortes en un directorio de trabajo temporal y, cuando se indique,
  subir las figuras definitivas a SeaweedFS mediante ``object_storage``.

No se depende de rutas absolutas compartidas entre contenedores: los recortes
se crean en directorios temporales locales y las figuras definitivas viven en
S3 con una clave estable.
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import pymupdf as fitz
from PIL import Image

from taygeta.core.schema import FigureReference
from taygeta.core.urls import relative_path_from_s3_uri

if TYPE_CHECKING:
    from taygeta.server.object_storage import ObjectStorage

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_DPI",
    "DEFAULT_PADDING",
    "MAX_AREA_RATIO",
    "NORMALIZED_MAX",
    "SUPPORTED_SUFFIXES",
    "CarriedContext",
    "FigureCrop",
    "ImageExtractionError",
    "PageRaster",
    "PixelRect",
    "assign_figure_ids",
    "carry_context",
    "denormalize_bbox",
    "detect_figures",
    "extract_and_save_figures",
    "crop_figure_bytes",
    "normalize_bbox",
    "open_document",
    "placeholder_for",
    "rasterize_document",
    "rasterize_page",
    "reading_order",
    "upload_crops",
    "working_directory",
]

#: Resolución de rasterizado por defecto, exigida por la especificación.
DEFAULT_DPI = 300

#: Unidades de la BBox normalizada.
NORMALIZED_MAX = 1000

#: Padding preventivo por defecto (1.5 % del tamaño de página).
DEFAULT_PADDING = 0.015

#: Fracción máxima del área de página que puede ocupar una figura. Un candidato
#: que cubre la página entera es, casi siempre, el fondo del documento o la
#: imagen de un escaneo, no un diagrama: se descarta para no capturar la hoja
#: completa como ``fig:N``.
MAX_AREA_RATIO = 0.6

#: Puntos por pulgada en el espacio de coordenadas de PDF.
POINTS_PER_INCH = 72.0

#: Extensiones de origen admitidas.
SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".pdf", ".png", ".jpg", ".jpeg"})


class ImageExtractionError(RuntimeError):
    """Error controlado durante la preparación o el recorte de figuras."""


class PixelRect(NamedTuple):
    """Rectángulo en coordenadas de píxel con origen en la esquina superior izquierda."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass(slots=True)
class PageRaster:
    """Página rasterizada a una resolución concreta."""

    page_index: int
    width_px: int
    height_px: int
    dpi: float
    png_bytes: bytes

    @property
    def size(self) -> tuple[int, int]:
        return (self.width_px, self.height_px)

    def save(self, path: str | Path) -> Path:
        """Guarda la página rasterizada en disco y devuelve la ruta."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.png_bytes)
        return destination


@dataclass(slots=True)
class FigureCrop:
    """Recorte de figura con sus coordenadas en varios espacios."""

    figure_id: str
    page_index: int
    normalized_bbox: list[int]
    clip_points: tuple[float, float, float, float]
    pixel_rect: PixelRect
    dpi: float
    path: Path | None = None
    asset_uri: str | None = None
    caption: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CarriedContext:
    """Fragmento incompleto de una página N que sirve de contexto para la N+1."""

    page_index: int
    text: str
    incomplete: bool = True
    placeholders: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Apertura y rasterización
# ---------------------------------------------------------------------------


def is_supported_source(source: str | Path) -> bool:
    """Indica si la extensión de ``source`` está admitida (PDF/PNG/JPG)."""
    return Path(source).suffix.lower() in SUPPORTED_SUFFIXES


@contextmanager
def open_document(source: str | Path) -> Iterator[fitz.Document]:
    """Abre un PDF o imagen con PyMuPDF y garantiza su cierre.

    Raises:
        ImageExtractionError: si el archivo no existe, la extensión no está
            admitida o PyMuPDF no puede abrirlo.
    """
    path = Path(source)
    if not path.is_file():
        raise ImageExtractionError(f"el archivo de origen no existe: {path}")
    if not is_supported_source(path):
        raise ImageExtractionError(
            f"extensión no admitida {path.suffix!r}; use una de {sorted(SUPPORTED_SUFFIXES)}"
        )
    try:
        document = fitz.open(path)
    except Exception as exc:  # PyMuPDF lanza tipos variados
        raise ImageExtractionError(f"no se pudo abrir {path}: {exc}") from exc
    try:
        yield document
    finally:
        document.close()


def _document_is_pdf(document: fitz.Document) -> bool:
    return bool(getattr(document, "is_pdf", False))


def _image_native_size(path: str | Path) -> tuple[int, int] | None:
    """Devuelve el tamaño en píxeles nativo de un archivo de imagen, si se puede."""
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:  # noqa: BLE001 - si falla, se usa la métrica de PyMuPDF
        return None


def rasterize_page(
    page: fitz.Page,
    *,
    dpi: int = DEFAULT_DPI,
    is_pdf: bool = True,
    zoom: float | None = None,
) -> PageRaster:
    """Rasteriza una página y devuelve sus bytes PNG.

    Para PDF se usa ``zoom = dpi / 72`` (300 DPI por defecto). Para imágenes se
    calibra el ``zoom`` de forma que se conserve la resolución nativa en píxeles.
    """
    if dpi <= 0:
        raise ImageExtractionError("el DPI debe ser positivo")
    if zoom is None:
        zoom = (dpi / POINTS_PER_INCH) if is_pdf else 1.0
    matrix = fitz.Matrix(zoom, zoom)
    try:
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    except Exception as exc:
        raise ImageExtractionError(f"no se pudo rasterizar la página {page.number}: {exc}") from exc
    return PageRaster(
        page_index=page.number,
        width_px=pixmap.width,
        height_px=pixmap.height,
        dpi=POINTS_PER_INCH * zoom,
        png_bytes=pixmap.tobytes("png"),
    )


def rasterize_document(source: str | Path, *, dpi: int = DEFAULT_DPI) -> list[PageRaster]:
    """Rasteriza todas las páginas de un documento manteniendo el índice físico."""
    with open_document(source) as document:
        is_pdf = _document_is_pdf(document)
        native = None if is_pdf else _image_native_size(source)
        pages = [
            rasterize_page(page, dpi=dpi, is_pdf=is_pdf, zoom=_page_zoom(page, is_pdf, dpi, native))
            for page in document
        ]
    logger.info("rasterizadas %d página(s) de %s", len(pages), source)
    return pages


def _page_zoom(
    page: fitz.Page,
    is_pdf: bool,
    dpi: int,
    native_size: tuple[int, int] | None,
) -> float:
    """Calcula el factor de escala para rasterizar una página sin perder detalle."""
    if is_pdf:
        return dpi / POINTS_PER_INCH
    if native_size and page.rect.width > 0:
        return native_size[0] / page.rect.width
    return 1.0


# ---------------------------------------------------------------------------
# Geometría: normalización y desnormalización
# ---------------------------------------------------------------------------


def denormalize_bbox(
    bbox: Sequence[int | float],
    page_width: float,
    page_height: float,
    *,
    padding: float = DEFAULT_PADDING,
    clamp: bool = True,
    as_int: bool = False,
) -> tuple[float, float, float, float]:
    """Convierte una BBox normalizada 0-1000 a coordenadas reales de página.

    La BBox tiene formato ``[ymin, xmin, ymax, xmax]``. ``page_width`` y
    ``page_height`` se expresan en la unidad destino (puntos de página). Se
    aplica un padding preventivo (por defecto el 1.5 % del tamaño de página) en
    los cuatro lados y, opcionalmente, se recorta al área de la página.

    Returns:
        Tupla ``(x0, y0, x1, y1)`` en la unidad de página.
    """
    if len(bbox) != 4:
        raise ImageExtractionError("la BBox debe tener cuatro valores [ymin, xmin, ymax, xmax]")
    if page_width <= 0 or page_height <= 0:
        raise ImageExtractionError("las dimensiones de página deben ser positivas")
    if padding < 0:
        raise ImageExtractionError("el padding no puede ser negativo")

    ymin, xmin, ymax, xmax = (float(value) for value in bbox)
    x0 = xmin / NORMALIZED_MAX * page_width
    x1 = xmax / NORMALIZED_MAX * page_width
    y0 = ymin / NORMALIZED_MAX * page_height
    y1 = ymax / NORMALIZED_MAX * page_height

    pad_x = padding * page_width
    pad_y = padding * page_height
    x0, x1 = x0 - pad_x, x1 + pad_x
    y0, y1 = y0 - pad_y, y1 + pad_y

    if clamp:
        x0 = min(max(x0, 0.0), page_width)
        x1 = min(max(x1, 0.0), page_width)
        y0 = min(max(y0, 0.0), page_height)
        y1 = min(max(y1, 0.0), page_height)

    if as_int:
        return (float(round(x0)), float(round(y0)), float(round(x1)), float(round(y1)))
    return (x0, y0, x1, y1)


def normalize_bbox(
    bbox: Sequence[int | float],
    page_width: float,
    page_height: float,
) -> list[int]:
    """Convierte una BBox en coordenadas de página a escala normalizada 0-1000."""
    if len(bbox) != 4:
        raise ImageExtractionError("la BBox debe tener cuatro valores")
    if page_width <= 0 or page_height <= 0:
        raise ImageExtractionError("las dimensiones de página deben ser positivas")
    x0, y0, x1, y1 = (float(value) for value in bbox)
    ymin = round(y0 / page_height * NORMALIZED_MAX)
    xmin = round(x0 / page_width * NORMALIZED_MAX)
    ymax = round(y1 / page_height * NORMALIZED_MAX)
    xmax = round(x1 / page_width * NORMALIZED_MAX)
    return [
        min(max(ymin, 0), NORMALIZED_MAX),
        min(max(xmin, 0), NORMALIZED_MAX),
        min(max(ymax, 0), NORMALIZED_MAX),
        min(max(xmax, 0), NORMALIZED_MAX),
    ]


def _rects_touch(a: fitz.Rect, b: fitz.Rect, gap: float) -> bool:
    return not (a.x1 + gap < b.x0 or b.x1 + gap < a.x0 or a.y1 + gap < b.y0 or b.y1 + gap < a.y0)


def _merge_rects(rects: list[fitz.Rect], gap: float) -> list[fitz.Rect]:
    """Une rectángulos que se solapan o quedan a menos de ``gap`` entre sí."""
    pending = list(rects)
    merged: list[fitz.Rect] = []
    while pending:
        current = pending.pop()
        changed = True
        while changed:
            changed = False
            remaining: list[fitz.Rect] = []
            for other in pending:
                if _rects_touch(current, other, gap):
                    current = current | other
                    changed = True
                else:
                    remaining.append(other)
            pending = remaining
        merged.append(current)
    return merged


# ---------------------------------------------------------------------------
# Detección de figuras y placeholders
# ---------------------------------------------------------------------------


def _is_rule_like(rect: fitz.Rect, page_rect: fitz.Rect) -> bool:
    """Indica si un candidato parece una regla/línea decorativa y no una figura.

    Las subrayados, tablas interiores o filos de página son rectángulos
    extremadamente finos y alargados que, fusionados, ensucian la detección:
    se descartan cuando ocupan al menos la mitad de la dimensión mayor de la
    página en una de sus dos direcciones.
    """
    if page_rect.width <= 0 or page_rect.height <= 0:
        return False
    horizontal = rect.height <= 2 and rect.width >= 0.5 * page_rect.width
    vertical = rect.width <= 2 and rect.height >= 0.5 * page_rect.height
    return horizontal or vertical


def detect_figures(
    page: fitz.Page,
    *,
    include_drawings: bool = True,
    min_area_ratio: float = 0.004,
    max_area_ratio: float = MAX_AREA_RATIO,
    max_figures: int = 50,
) -> list[FigureReference]:
    """Detecta figuras relevantes de una página y las devuelve normalizadas.

    Combina las imágenes embebidas (``get_images``/``get_image_rects``) con los
    dibujos vectoriales (``get_drawings``), descarta los elementos diminutos y
    las líneas decorativas y filtra los candidatos que cubren casi toda la
    página (fondos/escaneos), de modo que una hoja completa nunca se declare
    como ``fig:N``. Asigna identificadores ``fig:N`` en orden de lectura.
    """
    page_rect = page.rect
    page_area = max(page_rect.width * page_rect.height, 1.0)
    candidates: list[fitz.Rect] = []

    for image in page.get_images(full=True):
        xref = image[0]
        for rect in page.get_image_rects(xref):
            if not rect.is_empty:
                candidates.append(rect)

    if include_drawings:
        for drawing in page.get_drawings():
            rect = fitz.Rect(drawing.get("rect", fitz.EMPTY_RECT))
            if rect.is_empty or _is_rule_like(rect, page_rect):
                continue
            candidates.append(rect)

    merged = _merge_rects(candidates, gap=2.0)
    relevant = [
        rect
        for rect in merged
        if min_area_ratio <= (rect.width * rect.height) / page_area <= max_area_ratio
        and rect.width > 1
        and rect.height > 1
    ]
    relevant.sort(key=lambda rect: (round(rect.y0, 1), round(rect.x0, 1)))
    relevant = relevant[:max_figures]

    figures: list[FigureReference] = []
    for index, rect in enumerate(relevant):
        normalized = normalize_bbox(
            (rect.x0, rect.y0, rect.x1, rect.y1), page_rect.width, page_rect.height
        )
        ymin, xmin, ymax, xmax = normalized
        if ymin >= ymax or xmin >= xmax:
            continue
        figures.append(
            FigureReference(
                id=f"fig:{index + 1}",
                page=page.number,
                bbox=normalized,
            )
        )
    return figures


def reading_order(figures: Sequence[FigureReference]) -> list[FigureReference]:
    """Ordena las figuras por página y en flujo de lectura (arriba->abajo, izq->der)."""
    return sorted(
        figures,
        key=lambda figure: (
            figure.page,
            figure.bbox[0],
            figure.bbox[1],
        ),
    )


def assign_figure_ids(figures: Sequence[FigureReference]) -> list[FigureReference]:
    """Renumera las figuras en orden de lectura con identificadores ``fig:N``.

    Mantiene el resto de campos (página, bbox, caption, asset_path) intactos.
    """
    ordered = reading_order(figures)
    return [
        figure.model_copy(update={"id": f"fig:{index + 1}"}) for index, figure in enumerate(ordered)
    ]


def placeholder_for(figure_id: str) -> str:
    """Devuelve el placeholder de texto asociado a un identificador de figura."""
    return "{{" + figure_id + "}}"


def carry_context(previous_text: str | None, page_index: int) -> CarriedContext | None:
    """Conserva el fragmento incompleto de la página ``page_index`` como contexto.

    Devuelve ``None`` si no hay texto previo. El fragmento permite al pase OCR y
    al pase remoto interpretar ejercicios que continúan en la página siguiente.
    """
    if previous_text is None or not previous_text.strip():
        return None
    placeholders = [
        placeholder_for(f"fig:{token}") for token in _placeholder_numbers(previous_text)
    ]
    return CarriedContext(page_index=page_index, text=previous_text, placeholders=placeholders)


def _placeholder_numbers(text: str) -> list[str]:
    return re.findall(r"\{\{fig:(\d+)\}\}", text)


# ---------------------------------------------------------------------------
# Recorte de figuras
# ---------------------------------------------------------------------------


def _figure_spec(
    figure: FigureReference | Mapping[str, Any] | Sequence[int | float],
    index: int,
) -> tuple[str, list[int], str | None]:
    if isinstance(figure, FigureReference):
        return figure.id, list(figure.bbox), figure.caption
    if isinstance(figure, Mapping):
        figure_id = str(figure.get("id") or f"fig:{index + 1}")
        bbox = list(figure.get("bbox") or [])
        caption = figure.get("caption")
        return figure_id, [int(value) for value in bbox], caption
    bbox = [int(value) for value in figure]
    return f"fig:{index + 1}", bbox, None


def _safe_filename(figure_id: str, extension: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in figure_id)
    return f"{safe}{extension}"


def extract_and_save_figures(
    source: str | Path,
    page_index: int,
    figures: Sequence[FigureReference | Mapping[str, Any] | Sequence[int | float]],
    output_dir: str | Path,
    *,
    dpi: int = DEFAULT_DPI,
    padding: float = DEFAULT_PADDING,
    image_format: str = "png",
) -> list[FigureCrop]:
    """Recorta y guarda las figuras indicadas en ``output_dir``.

    Desnormaliza cada BBox 0-1000 al tamaño real de la página añadiendo el
    padding preventivo, rasteriza el recorte a ``dpi`` y escribe el archivo
    temporal. Devuelve las coordenadas normalizadas, el rectángulo en puntos y
    el rectángulo en píxeles para asociar la figura con el enunciado exacto.

    Args:
        source: Ruta del PDF o imagen.
        page_index: Índice físico de página basado en cero.
        figures: Figuras con BBox normalizada (``FigureReference``, mapping o
            secuencia ``[ymin, xmin, ymax, xmax]``).
        output_dir: Directorio temporal donde escribir los recortes.
        dpi: Resolución de rasterizado del recorte.
        padding: Padding preventivo como fracción del tamaño de página.
        image_format: Formato de imagen de salida (por defecto ``png``).
    """
    extension = f".{image_format.lstrip('.')}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open_document(source) as document:
        if page_index < 0 or page_index >= document.page_count:
            raise ImageExtractionError(
                f"índice de página {page_index} fuera de rango (0..{document.page_count - 1})"
            )
        page = document.load_page(page_index)
        is_pdf = _document_is_pdf(document)
        page_rect = page.rect
        native = None if is_pdf else _image_native_size(source)
        zoom = _page_zoom(page, is_pdf, dpi, native)
        matrix = fitz.Matrix(zoom, zoom)
        effective_dpi = POINTS_PER_INCH * zoom

        crops: list[FigureCrop] = []
        for index, figure in enumerate(figures):
            figure_id, bbox, caption = _figure_spec(figure, index)
            x0, y0, x1, y1 = denormalize_bbox(
                bbox, page_rect.width, page_rect.height, padding=padding
            )
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                logger.warning("se omite la figura %s: BBox degenerada %s", figure_id, bbox)
                continue
            clip = fitz.Rect(x0, y0, x1, y1)
            try:
                pixmap = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
            except Exception as exc:
                raise ImageExtractionError(
                    f"no se pudo recortar la figura {figure_id} en la página {page_index}: {exc}"
                ) from exc

            destination = output_path / _safe_filename(figure_id, extension)
            pixmap.save(str(destination))
            pixel_rect = PixelRect(
                x0=round(x0 * zoom),
                y0=round(y0 * zoom),
                x1=round(x1 * zoom),
                y1=round(y1 * zoom),
            )
            crops.append(
                FigureCrop(
                    figure_id=figure_id,
                    page_index=page_index,
                    normalized_bbox=[int(round(value)) for value in bbox],
                    clip_points=(x0, y0, x1, y1),
                    pixel_rect=pixel_rect,
                    dpi=effective_dpi,
                    path=destination,
                    caption=caption,
                    metadata={
                        "padding": padding,
                        "page_width": page_rect.width,
                        "page_height": page_rect.height,
                    },
                )
            )
            logger.debug(
                "figura %s recortada en página %d -> %s", figure_id, page_index, destination
            )
    return crops


def crop_figure_bytes(
    source: str | Path,
    page_index: int,
    bbox: Sequence[int],
    *,
    dpi: int = DEFAULT_DPI,
    padding: float = DEFAULT_PADDING,
) -> bytes:
    """Recorta una región normalizada de una página y devuelve sus bytes PNG.

    Usa la misma geometría que :func:`extract_and_save_figures` (padding
    preventivo, desnormalización y rasterización del clip), pero devuelve los
    bytes en memoria: así la UI puede sobrescribir un asset en S3 sin tocar el
    sistema de archivos ni dejar copias temporales.

    Raises:
        ImageExtractionError: si la página está fuera de rango, la BBox es
            degenerada o el recorte falla.
    """
    with open_document(source) as document:
        if page_index < 0 or page_index >= document.page_count:
            raise ImageExtractionError(
                f"índice de página {page_index} fuera de rango (0..{document.page_count - 1})"
            )
        page = document.load_page(page_index)
        is_pdf = _document_is_pdf(document)
        page_rect = page.rect
        native = None if is_pdf else _image_native_size(source)
        zoom = _page_zoom(page, is_pdf, dpi, native)
        x0, y0, x1, y1 = denormalize_bbox(bbox, page_rect.width, page_rect.height, padding=padding)
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            raise ImageExtractionError(f"BBox degenerada al recortar: {bbox}")
        try:
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(zoom, zoom), clip=fitz.Rect(x0, y0, x1, y1), alpha=False
            )
        except Exception as exc:  # noqa: BLE001 - PyMuPDF lanza tipos variados
            raise ImageExtractionError(
                f"no se pudo recortar la figura de la página {page_index}: {exc}"
            ) from exc
    return pixmap.tobytes("png")


def upload_crops(
    crops: Sequence[FigureCrop],
    storage: ObjectStorage,
    *,
    prefix: str = "figures",
    delete_local: bool = True,
    extension: str = ".png",
) -> list[FigureCrop]:
    """Sube los recortes a SeaweedFS y anota su ruta relativa estable.

    Las figuras definitivas viven en S3 y se referencian mediante una ruta
    relativa ``/buckets/{bucket}/{key}`` (forma canónica de almacenamiento,
    independiente del endpoint). Cuando ``delete_local`` es ``True`` se elimina
    la copia temporal para no guardar permanentemente una segunda copia local.
    """
    for crop in crops:
        if crop.path is None:
            continue
        uri = storage.store_asset_file(crop.path, prefix=prefix, asset_id=None, extension=extension)
        crop.asset_uri = relative_path_from_s3_uri(uri)
        if delete_local:
            try:
                crop.path.unlink()
            except FileNotFoundError:
                pass
            crop.path = None
    return list(crops)


# ---------------------------------------------------------------------------
# Directorio de trabajo temporal
# ---------------------------------------------------------------------------


@contextmanager
def working_directory(prefix: str = "taygeta-ingest-") -> Iterator[Path]:
    """Crea un directorio de trabajo temporal y lo elimina al salir.

    No toca la fuente subida por el usuario ni los objetos persistentes.
    """
    path = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
