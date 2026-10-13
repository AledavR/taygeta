"""Interfaz Human-in-the-Loop de Taygeta (sección 8).

Aplicación Streamlit para revisar la ingesta híbrida:

* Subida de PDF/PNG/JPG al directorio ``UPLOADS_DIR`` y procesamiento con el
  pipeline de doble pase, mostrando el estado.
* Panel de revisión con visualización **en paralelo** del enunciado Markdown y
  de la imagen original (recorte de figura desde SeaweedFS o página rasterizada).
* Estados ``VALID`` / ``NEEDS_ATTENTION`` con explicación técnica.
* Campo de reintento con contexto correctivo que reanaliza el enunciado.
* Edición manual del Markdown y de los campos, slider de dificultad, selectores
  de curso/tema/subtema y multiselect de prerrequisitos con alta dinámica.
* Botón de confirmación que persiste las preguntas **aprobadas** en ``taygeta.db``
  y garantiza que las figuras estén en el bucket persistente de SeaweedFS.

La corrección manual prevalece sobre la transcripción automática. Ninguna
pregunta se guarda en la base definitiva sin la confirmación explícita del
usuario, y un fallo de OCR o de validación nunca se oculta marcando la pregunta
como válida.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

from taygeta.agents.embeddings import Embedder, question_embedding_text
from taygeta.core.schema import (
    Choice,
    EnrichedQuestion,
    FigureReference,
    QuestionRaw,
    TranscriptionIntermediate,
    get_public_url,
    normalize_slug,
    sanitize_math_delimiters,
    slugify,
    utc_now_iso,
)
from taygeta.core.settings import Settings, get_settings
from taygeta.core.urls import build_relative_path, split_asset_reference
from taygeta.domain.taxonomy import format_course_name
from taygeta.ingest.image_extractor import (
    DEFAULT_DPI,
    PageRaster,
    crop_figure_bytes,
    is_supported_source,
    open_document,
    rasterize_page,
)
from taygeta.ingest.pipeline import IngestPipeline, StagedQuestion, StagingResult
from taygeta.ingest.validator import STATUS_NEEDS_ATTENTION, STATUS_VALID
from taygeta.server.db import Database, DatabaseError
from taygeta.server.object_storage import ObjectStorage, ObjectStorageError
from taygeta.ui.figure_editing import (
    append_figure_placeholder,
    bbox_to_pixel_box,
    next_figure_id,
    pixel_box_to_normalized_bbox,
    reconcile_figures_with_statement,
    strip_figure_placeholder,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ConfirmResult",
    "build_enriched_question",
    "figure_bytes",
    "list_uploads",
    "main",
    "page_image_bytes",
    "persist_approved_questions",
    "public_asset_url",
    "public_s3_url",
    "sanitize_upload_name",
    "save_upload",
]

#: Extensiones aceptadas por el cargador de archivos.
UPLOAD_EXTENSIONS = ["pdf", "png", "jpg", "jpeg"]

_SLUG_RE = re.compile(r"^[a-z0-9_]+$")

#: Alias locales de los helpers compartidos de ``taygeta.core.schema``.
_slugify = slugify
_normalize_slug = normalize_slug


def _slug_split(text: str | None) -> list[str]:
    """Divide una cadena separada por comas en slugs normalizados (sin vacíos)."""
    if not text:
        return []
    return [slug for slug in (_slugify(part) for part in text.split(",")) if slug]


# ---------------------------------------------------------------------------
# Aislamiento del estado de staging (UUID)
# ---------------------------------------------------------------------------

#: Claves de ``st.session_state`` que contienen preguntas en staging y su origen.
_STAGING_STATE_KEYS = ("staging_result", "processed_path", "staged_questions", "_orphan_assets")

#: Prefijos de los controles de revisión (widgets) asociados a cada pregunta.
_STAGING_WIDGET_PREFIXES = (
    "statement_",
    "qtype_",
    "choice_id_",
    "choice_text_",
    "course_",
    "topic_",
    "subtopic_",
    "diff_",
    "prereq_",
    "new_prereq_",
    "hint_",
    "approve_",
    "refine_",
    "fig_",
)


def _clear_staged_questions() -> None:
    """Destruye el estado de preguntas en staging y sus controles de la sesión.

    Se invoca al detectar un cambio de documento o un nuevo ``session_id`` en la
    URL, de modo que la revisión nunca mezcle preguntas de documentos o sesiones
    distintas (aislamiento de UUID).
    """
    for key in _STAGING_STATE_KEYS:
        st.session_state.pop(key, None)
    for key in list(st.session_state.keys()):
        if key.startswith(_STAGING_WIDGET_PREFIXES):
            del st.session_state[key]


def _guard_staging_source(source: str | None) -> None:
    """Limpia el staging si el documento a revisar cambió respecto al anterior."""
    if source != st.session_state.get("processed_path"):
        _clear_staged_questions()


def _sync_url_session() -> None:
    """Detecta un nuevo ``session_id`` en la URL y aísla el staging previo."""
    sid = st.query_params.get("session_id")
    if sid != st.session_state.get("_url_session_id"):
        _clear_staged_questions()
        st.session_state["_url_session_id"] = sid


# ---------------------------------------------------------------------------
# Utilidades de archivos y figuras (testables sin Streamlit)
# ---------------------------------------------------------------------------


def sanitize_upload_name(name: str) -> str:
    """Sanea el nombre de un archivo subido, evitando traversal y caracteres raros."""
    base = Path(name).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", base).lstrip(".")
    return cleaned or "upload"


def save_upload(name: str, data: bytes, uploads_dir: str | Path) -> Path:
    """Guarda los bytes subidos en ``uploads_dir`` y devuelve la ruta final."""
    directory = Path(uploads_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / sanitize_upload_name(name)
    if destination.exists():
        destination = directory / f"{destination.stem}_{uuid.uuid4().hex[:8]}{destination.suffix}"
    destination.write_bytes(data)
    return destination


def list_uploads(uploads_dir: str | Path) -> list[Path]:
    """Lista los documentos admitidos presentes en ``uploads_dir``."""
    directory = Path(uploads_dir)
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and is_supported_source(path)),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def figure_bytes(storage: ObjectStorage | None, asset_path: str | None) -> bytes | None:
    """Recupera los bytes de una figura desde SeaweedFS a partir de su referencia.

    Acepta ``/buckets/{bucket}/{key}``, ``s3://bucket/key`` o una clave de objeto.
    Devuelve ``None`` si la referencia está vacía o el objeto no se puede recuperar.
    """
    if storage is None or not asset_path:
        return None
    try:
        bucket, key = split_asset_reference(asset_path)
        if bucket:
            return storage.get_bytes(key, bucket=bucket)
        return storage.get_bytes(key)
    except (ObjectStorageError, ValueError) as exc:
        logger.warning("no se pudo recuperar la figura %s: %s", asset_path, exc)
        return None


def page_image_bytes(source_path: str | Path, page_index: int, *, dpi: int = 150) -> bytes | None:
    """Rasteriza una página del documento origen para mostrarla en la revisión."""
    try:
        with open_document(source_path) as document:
            if page_index < 0 or page_index >= document.page_count:
                return None
            page = document.load_page(page_index)
            is_pdf = bool(getattr(document, "is_pdf", False))
            return rasterize_page(page, dpi=dpi, is_pdf=is_pdf).png_bytes
    except Exception as exc:  # noqa: BLE001 - la vista previa nunca debe romper la UI
        logger.warning("no se pudo rasterizar la página %s: %s", page_index, exc)
        return None


def page_image_raster(
    source_path: str | Path, page_index: int, *, dpi: int = 150
) -> PageRaster | None:
    """Devuelve la ``PageRaster`` de una página para el cropper (bytes + tamaño).

    ``PageRaster`` expone ``png_bytes`` y ``width_px``/``height_px``, que se
    necesitan tanto para alimentar el recorte manual como para convertir el
    recuadro de píxeles a la BBox normalizada.
    """
    try:
        with open_document(source_path) as document:
            if page_index < 0 or page_index >= document.page_count:
                return None
            page = document.load_page(page_index)
            is_pdf = bool(getattr(document, "is_pdf", False))
            return rasterize_page(page, dpi=dpi, is_pdf=is_pdf)
    except Exception as exc:  # noqa: BLE001 - la vista previa nunca debe romper la UI
        logger.warning("no se pudo rasterizar la página %s: %s", page_index, exc)
        return None


def document_page_count(source_path: str | Path | None) -> int | None:
    """Devuelve el número de páginas del documento origen, o ``None`` si es inaccesible.

    Se usa para ofrecer el selector de página del recorte manual de figuras.
    """
    if source_path is None:
        return None
    try:
        with open_document(source_path) as document:
            return document.page_count
    except Exception as exc:  # noqa: BLE001 - la UI nunca debe romper por el origen
        logger.warning("no se pudo contar las páginas de %s: %s", source_path, exc)
        return None


# ---------------------------------------------------------------------------
# Edición de figuras en la revisión (estado de sesión)
# ---------------------------------------------------------------------------


def _figures_state(index: int, default: Sequence[Any]) -> list[dict[str, Any]]:
    """Devuelve las figuras editables de la pregunta ``index`` como dicts.

    Siembra ``st.session_state["figures_{index}"]`` desde ``default`` (las
    figuras del enunciado estructurado) la primera vez; a partir de ahí, la
    lista en sesión es la fuente de verdad que refleja borrados y recortes.
    """
    key = f"figures_{index}"
    if key not in st.session_state:
        st.session_state[key] = [
            figure if isinstance(figure, dict) else figure.model_dump(mode="json")
            for figure in default
        ]
    return st.session_state[key]


def _reset_figures_state(index: int, figures: Sequence[Any]) -> None:
    """Reemplaza el estado de figuras de la pregunta ``index`` tras reanalizar."""
    st.session_state[f"figures_{index}"] = [
        figure if isinstance(figure, dict) else figure.model_dump(mode="json") for figure in figures
    ]


def _register_orphan(question_id: str, asset_path: str | None) -> None:
    """Registra un asset de figura como huérfano para purgarlo al confirmar.

    El borrado físico se difiere a la confirmación de la pregunta para no
    destruir el objeto si el usuario abandona la revisión.
    """
    if not asset_path:
        return
    orphans = st.session_state.setdefault("_orphan_assets", {})
    orphans.setdefault(question_id, set()).add(asset_path)


def _delete_asset(storage: ObjectStorage | None, asset_path: str) -> None:
    """Borra un asset por su referencia estable (best-effort, nunca bloquea)."""
    if storage is None:
        return
    try:
        bucket, key = split_asset_reference(asset_path, default_bucket=storage.bucket)
        storage.delete(key, bucket=bucket or None)
    except (ObjectStorageError, ValueError) as exc:
        logger.warning("no se pudo purgar el asset huérfano %s: %s", asset_path, exc)


def _purge_orphan_assets(confirmed_ids: set[str], storage: ObjectStorage | None) -> None:
    """Borra los assets huérfanos de las preguntas efectivamente confirmadas."""
    orphans = st.session_state.pop("_orphan_assets", {})
    for question_id in confirmed_ids:
        for asset_path in orphans.get(question_id, set()):
            _delete_asset(storage, asset_path)


def _remove_figure(index: int, figure_id: str) -> None:
    """Elimina una figura del estado y limpia su placeholder del enunciado.

    Extrae la figura del estado de sesión de la pregunta ``index`` y elimina
    automáticamente todos los ``{{fig:N}}`` correspondientes del enunciado
    (``statement_{index}``), conservando la correspondencia uno a uno que exige
    ``QuestionRaw``. Registra el asset como huérfano para purgarlo al confirmar.
    """
    key = f"figures_{index}"
    figures = st.session_state.get(key, [])
    removed = [figure for figure in figures if figure.get("id") == figure_id]
    st.session_state[key] = [figure for figure in figures if figure.get("id") != figure_id]
    statement_key = f"statement_{index}"
    current = st.session_state.get(statement_key)
    if isinstance(current, str):
        st.session_state[statement_key] = strip_figure_placeholder(current, figure_id)

    result = st.session_state.get("staging_result")
    question = (
        result.questions[index].question
        if result is not None and 0 <= index < len(result.questions)
        else None
    )
    question_id = getattr(question, "id", None)
    for figure in removed:
        _register_orphan(str(question_id), figure.get("asset_path"))


def _replace_figure_crop(
    index: int,
    figure_id: str,
    new_bbox: list[int],
    source_path: str | None,
    storage: ObjectStorage | None,
) -> None:
    """Re-recorta una figura y sobrescribe su asset en el almacenamiento.

    Recorta la región indicada del documento origen a 300 DPI y sobrescribe el
    objeto existente en SeaweedFS (misma clave), de modo que ``asset_path`` no
    cambia. Actualiza el estado de sesión con la nueva BBox para refrescar la
    vista previa en el siguiente render.

    Raises:
        ValueError: si la figura no tiene clave de asset o el origen no permite
            recortar.
        ImageExtractionError: si el recorte falla.
    """
    if source_path is None:
        raise ValueError("no hay documento de origen disponible para recortar")
    figures = st.session_state.get(f"figures_{index}", [])
    updated: list[dict[str, Any]] = []
    for figure in figures:
        if figure.get("id") != figure_id:
            updated.append(figure)
            continue
        asset_path = figure.get("asset_path")
        if not asset_path:
            raise ValueError(
                f"la figura {figure_id} no tiene asset persistente; no se puede sobrescribir"
            )
        bucket, key = split_asset_reference(asset_path, default_bucket=storage.bucket)
        png = crop_figure_bytes(
            source_path,
            int(figure.get("page", 0)),
            new_bbox,
            dpi=DEFAULT_DPI,
        )
        if storage is None:
            raise ValueError("almacenamiento no disponible para guardar el recorte")
        storage.put_bytes(key, png, content_type="image/png", bucket=bucket or None)
        updated.append({**figure, "bbox": list(new_bbox)})
    st.session_state[f"figures_{index}"] = updated


def _add_new_figure(
    index: int,
    page_index: int,
    new_bbox: list[int],
    source_path: str | None,
    storage: ObjectStorage | None,
) -> str:
    """Recorta, sube y registra una figura nueva en el estado de la pregunta.

    Cubre los falsos negativos: el usuario recorta una región de la página
    original y este flujo la persiste con la misma convención del pipeline
    (``figures/{uuid}.png``), asigna el siguiente ``fig:N`` libre de la pregunta
    y añade su placeholder al final del enunciado editado para que el usuario lo
    mueva donde corresponda.

    Raises:
        ValueError: si no hay origen o almacenamiento disponibles.
        ImageExtractionError: si el recorte falla.
        ObjectStorageError: si la subida del asset falla.
    """
    if source_path is None:
        raise ValueError("no hay documento de origen disponible para recortar")
    if storage is None:
        raise ValueError("almacenamiento no disponible para guardar la figura")
    png = crop_figure_bytes(source_path, page_index, new_bbox, dpi=DEFAULT_DPI)
    uri = storage.store_asset_bytes(
        png,
        prefix="figures",
        asset_id=None,
        extension=".png",
        content_type="image/png",
    )
    asset_path = storage.relative_path(uri)

    figures = st.session_state.get(f"figures_{index}", [])
    statement = st.session_state.get(f"statement_{index}", "")
    if not isinstance(statement, str):
        statement = ""
    figure_id = next_figure_id(list(figures), statement)
    figures.append(
        {
            "id": figure_id,
            "page": page_index,
            "bbox": list(new_bbox),
            "caption": None,
            "asset_path": asset_path,
        }
    )
    st.session_state[f"figures_{index}"] = figures
    st.session_state[f"statement_{index}"] = append_figure_placeholder(statement, figure_id)
    return figure_id


def public_s3_url(url: str | None, settings: Settings) -> str:
    """Reescribe el endpoint interno de S3 por el público de la UI.

    Sustituye el prefijo ``S3_ENDPOINT_URL`` (p. ej. ``http://seaweedfs-s3:8333``)
    por ``S3_PUBLIC_ENDPOINT`` (p. ej. ``http://localhost:8333``) para poder abrir
    enlaces e imágenes desde el navegador del host. Si no hay endpoint público o
    la URL no usa el interno, se devuelve sin cambios.
    """
    if not url:
        return ""
    internal = (settings.S3_ENDPOINT_URL or "").rstrip("/")
    public = (settings.S3_PUBLIC_ENDPOINT or "").rstrip("/")
    if public and internal and url.startswith(internal):
        return public + url[len(internal) :]
    return url


def public_asset_url(asset_path: str | None, settings: Settings) -> str:
    """Compone la URL pública de una figura con ``get_public_url`` de schema.

    Convierte cualquier referencia de asset (``/buckets/{bucket}/{key}``,
    ``s3://bucket/key`` o clave relativa) a su ruta relativa canónica y la
    concatena con ``S3_PUBLIC_BASE_URL``. Así la UI renderiza imágenes con la URL
    pública del host, nunca con URLs internas de Docker (p. ej.
    ``http://seaweedfs-s3:8333``).
    """
    if not asset_path:
        return ""
    try:
        bucket, key = split_asset_reference(asset_path, default_bucket=settings.S3_BUCKET)
        relative = build_relative_path(bucket, key)
    except ValueError:
        return ""
    return get_public_url(relative)


# ---------------------------------------------------------------------------
# Construcción y persistencia de preguntas aprobadas
# ---------------------------------------------------------------------------


def build_enriched_question(
    base: QuestionRaw,
    *,
    course: str,
    topic: str,
    subtopic: str | None,
    difficulty: int,
    prerequisites: Sequence[str],
    source_metadata: dict[str, Any] | None = None,
    figures: Sequence[FigureReference] | None = None,
) -> EnrichedQuestion:
    """Compone una :class:`EnrichedQuestion` a partir de los campos revisados.

    Los valores editados manualmente prevalecen sobre la transcripción
    automática. ``course``, ``topic`` y ``subtopic`` se normalizan a slugs
    válidos (``^[a-z0-9_]+$``) para evitar ``ValidationError``; si ``course`` o
    ``topic`` quedan vacíos se usa ``"general"`` y el ``subtopic`` vacío se
    convierte en ``None``. Los prerrequisitos se normalizan igual y se descartan
    los que queden vacíos. ``figures`` permite sustituir las figuras de
    ``base.figures`` con la selección editada en la revisión (borrados y
    recortes); si se omite, se conservan las originales.
    """
    return EnrichedQuestion(
        id=base.id,
        statement_raw=base.statement_raw,
        question_type=base.question_type,
        choices=base.choices,
        figures=figures if figures is not None else base.figures,
        notation_notes=base.notation_notes,
        course=_normalize_slug(course),
        topic=_normalize_slug(topic),
        subtopic=_slugify(subtopic) or None,
        difficulty=difficulty,
        prerequisites=[slug for slug in (_slugify(item) for item in prerequisites) if slug],
        source_metadata=source_metadata or {},
        created_at=utc_now_iso(),
    )


@dataclass(slots=True)
class ConfirmResult:
    """Resultado de la confirmación y persistencia de preguntas."""

    saved: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _missing_figure_assets(question: EnrichedQuestion, storage: ObjectStorage | None) -> list[str]:
    missing: list[str] = []
    for figure in question.figures:
        if not figure.asset_path:
            missing.append(f"{figure.id} (sin asset_path)")
            continue
        if storage is None:
            continue
        try:
            bucket, key = split_asset_reference(figure.asset_path)
            if bucket:
                if not storage.exists(key, bucket=bucket):
                    missing.append(figure.asset_path)
            elif not storage.exists(key):
                missing.append(figure.asset_path)
        except (ObjectStorageError, ValueError):
            missing.append(figure.asset_path)
    return missing


def persist_approved_questions(
    questions: Sequence[EnrichedQuestion],
    database: Database,
    storage: ObjectStorage | None = None,
    *,
    session_id: str | None = None,
    embedder: Embedder | None = None,
) -> ConfirmResult:
    """Persiste las preguntas aprobadas tras verificar sus figuras en el bucket.

    Usa ``upsert_question`` para que la confirmación sea idempotente. Solo marca
    la sesión de staging como ``CONFIRMED`` cuando ha guardado alguna pregunta.
    Genera (best-effort) el embedding de cada pregunta para la deduplicación
    semántica posterior; si el proveedor no está disponible, se omite sin
    bloquear la confirmación.
    """
    if embedder is None:
        embedder = Embedder(get_settings())
    result = ConfirmResult()
    for question in questions:
        try:
            missing = _missing_figure_assets(question, storage)
            if missing:
                result.errors.append(
                    f"{question.id}: figuras no presentes en el bucket: {', '.join(missing)}"
                )
                continue
            database.add_prerequisites(list(question.prerequisites))
            database.upsert_question(question, embedding=_embed_or_none(embedder, question))
            result.saved.append(question.id)
        except Exception as exc:  # noqa: BLE001 - se informa al usuario, no se oculta
            result.errors.append(f"{question.id}: {exc}")

    if session_id and result.saved:
        try:
            database.update_staging_session(session_id, status="CONFIRMED")
        except DatabaseError as exc:
            result.errors.append(f"no se pudo marcar la sesión como confirmada: {exc}")
    return result


def _embed_or_none(embedder: Embedder, question: EnrichedQuestion) -> list[float] | None:
    """Genera el embedding de una pregunta sin elevar ante fallos del proveedor."""
    try:
        return embedder.embed(question_embedding_text(question))
    except Exception:  # noqa: BLE001 - la deduplicación semántica es opcional
        logger.warning("no se pudo generar el embedding de la pregunta %s", question.id)
        return None


# ---------------------------------------------------------------------------
# Recursos cacheados
# ---------------------------------------------------------------------------


@st.cache_resource
def _resources() -> tuple[Settings, Database, ObjectStorage, IngestPipeline]:
    settings = get_settings()
    database = Database(settings=settings)
    storage = ObjectStorage(settings)
    pipeline = IngestPipeline(settings, storage=storage, database=database)
    return settings, database, storage, pipeline


# ---------------------------------------------------------------------------
# Renderizado
# ---------------------------------------------------------------------------


def _cache_busted_public_url(asset_path: str | None, settings: Settings, seed: Any = "") -> str:
    """URL pública de una figura con parámetro de versión anti-caché.

    La vista previa lee los bytes directamente de SeaweedFS (sin caché de
    navegador); el enlace externo añade ``?v=`` derivado del asset y de la BBox
    para que el navegador no sirva una versión antigua tras un recorte.
    """
    url = public_asset_url(asset_path, settings)
    if not url:
        return ""
    token = hashlib.sha256(f"{asset_path}|{seed}".encode()).hexdigest()[:8]
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}v={token}"


def _render_figure_cropper(
    index: int,
    figure: dict[str, Any],
    source_path: str | None,
    storage: ObjectStorage | None,
) -> None:
    """Despliega el recorte manual (bounding box) para una figura concreta."""
    figure_id = figure.get("id") or "fig:?"
    page_index = int(figure.get("page", 0))
    bbox = figure.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        bbox = [0, 0, 0, 0]
    with st.expander(f"Recortar manualmente — {figure_id}"):
        raster = page_image_raster(source_path, page_index)
        if raster is None:
            st.caption("No hay documento de origen para recortar esta figura.")
            return
        buffer = io.BytesIO(raster.png_bytes)
        image = Image.open(buffer)
        image.load()
        crop_key = f"fig_crop_{index}_{figure_id}"
        default_coords = bbox_to_pixel_box(bbox, raster.width_px, raster.height_px)
        try:
            box = st_cropper(
                image,
                realtime_update=True,
                default_coords=default_coords,
                return_type="box",
                key=crop_key,
            )
        except Exception as exc:  # noqa: BLE001 - el cropper nunca debe romper la revisión
            st.error(f"No se pudo inicializar el recorte: {exc}")
            return
        if st.button("Aplicar recorte", key=f"fig_crop_apply_{index}_{figure_id}"):
            try:
                new_bbox = pixel_box_to_normalized_bbox(box, raster.width_px, raster.height_px)
                _replace_figure_crop(index, figure_id, new_bbox, source_path, storage)
            except Exception as exc:  # noqa: BLE001 - se informa al usuario, no se oculta
                st.error(f"No se pudo aplicar el recorte de {figure_id}: {exc}")
                return
            st.success(f"{figure_id} recortada y guardada.")
            st.rerun()


def _render_add_figure(
    index: int,
    source_path: str | None,
    storage: ObjectStorage | None,
) -> None:
    """Permite añadir manualmente una figura omitida por el modelo (falso negativo).

    El botón abre un recorte sobre la página original completa del documento;
    al confirmar, el recorte se sube a SeaweedFS con la convención del pipeline,
    se asigna el siguiente ``fig:N`` libre de la pregunta y se inserta su
    placeholder al final del enunciado en el editor Markdown (refrescando la
    vista previa). El flag ``fig_add_open_{index}`` queda abierto para poder
    añadir varias figuras sin tener que volver a pulsar el botón.
    """
    open_key = f"fig_add_open_{index}"
    if st.button("➕ Añadir nueva figura", key=f"fig_add_{index}"):
        st.session_state[open_key] = True
        st.rerun()
    if not st.session_state.get(open_key, False):
        return

    page_count = document_page_count(source_path)
    if page_count is None or page_count <= 0:
        st.caption("No hay documento de origen para añadir figuras.")
        return
    default_page = 0
    figures = st.session_state.get(f"figures_{index}", [])
    if figures:
        default_page = min(int(figures[0].get("page", 0)), page_count - 1)
    page_index = st.selectbox(
        "Página original del documento",
        options=list(range(page_count)),
        index=default_page,
        key=f"fig_add_page_{index}",
    )
    raster = page_image_raster(source_path, page_index)
    if raster is None:
        st.caption("No se pudo cargar la página original para recortar.")
        return
    buffer = io.BytesIO(raster.png_bytes)
    image = Image.open(buffer)
    image.load()
    crop_key = f"fig_add_crop_{index}_{page_index}"
    try:
        box = st_cropper(
            image,
            realtime_update=True,
            default_coords=None,
            return_type="box",
            key=crop_key,
        )
    except Exception as exc:  # noqa: BLE001 - el cropper nunca debe romper la revisión
        st.error(f"No se pudo inicializar el recorte: {exc}")
        return
    if st.button("Añadir figura", key=f"fig_add_apply_{index}"):
        try:
            new_bbox = pixel_box_to_normalized_bbox(box, raster.width_px, raster.height_px)
            figure_id = _add_new_figure(index, page_index, new_bbox, source_path, storage)
        except Exception as exc:  # noqa: BLE001 - se informa al usuario, no se oculta
            st.error(f"No se pudo añadir la figura: {exc}")
            return
        st.success(f"{figure_id} añadida; su placeholder quedó al final del enunciado.")
        st.rerun()


def _render_figure_review(
    index: int,
    staged: StagedQuestion,
    source_path: str | None,
    storage: ObjectStorage | None,
    settings: Settings,
) -> None:
    """Muestra las figuras de la pregunta con borrado rápido y recorte manual.

    La lista de figuras del estado de sesión (``figures_{index}``) es la fuente
    de verdad durante la revisión: refleja los borrados, recortes y altas. Por
    figura se muestra el recorte (bytes frescos de SeaweedFS), un botón para
    eliminarla (que limpia también su ``{{fig:N}}`` del enunciado) y un recorte
    manual con el cropper sobre la página original del documento. Debajo, el
    botón ``➕ Añadir nueva figura`` permite cubrir falsos negativos recortando
    figuras que el modelo de visión omitió por completo.
    """
    question = staged.question
    figures = _figures_state(index, question.figures if question is not None else [])
    shown = False
    for figure in figures:
        figure_id = figure.get("id") or "fig:?"
        page_index = int(figure.get("page", 0))
        asset_path = figure.get("asset_path")
        bbox = figure.get("bbox") or [0, 0, 0, 0]
        data = figure_bytes(storage, asset_path)
        public_url = _cache_busted_public_url(asset_path, settings, seed=bbox)
        if data:
            st.image(data, caption=f"{figure_id} — recorte (página {page_index})")
            shown = True
        elif public_url:
            st.image(public_url, caption=f"{figure_id} — recorte (página {page_index})")
            shown = True
        if public_url:
            st.caption(f"[Abrir {figure_id} en el navegador]({public_url})")
        if st.button("Eliminar figura", key=f"fig_delete_{index}_{figure_id}", type="secondary"):
            _remove_figure(index, figure_id)
            st.rerun()
        _render_figure_cropper(index, figure, source_path, storage)
    _render_add_figure(index, source_path, storage)
    if shown or source_path is None:
        return
    pages = [0]
    if question is not None and question.figures:
        pages = sorted({figure.page for figure in question.figures})
    for page_index in pages:
        data = page_image_bytes(source_path, page_index)
        if data:
            st.image(data, caption=f"Página original {page_index}")


def _taxonomy_defaults(staged: StagedQuestion) -> dict[str, Any]:
    """Valores por defecto seguros (vía ``getattr``) para la taxonomía.

    El objeto en staging es un :class:`EnrichedQuestion` (el estructurador infiere
    la taxonomía en el pase único), por lo que ``course``, ``topic``, ``subtopic``,
    ``difficulty`` y ``prerequisites`` ya vienen rellenos. Se accede con
    ``getattr`` por robustez ante preguntas no estructuradas (``question=None``).
    """
    base = staged.question
    course = getattr(base, "course", "") or ""
    topic = getattr(base, "topic", "") or ""
    subtopic = getattr(base, "subtopic", None)
    difficulty = getattr(base, "difficulty", 3)
    prerequisites = getattr(base, "prerequisites", None) or []

    if not isinstance(subtopic, str) or not subtopic:
        subtopic = None
    if not isinstance(difficulty, int) or isinstance(difficulty, bool) or not 1 <= difficulty <= 5:
        difficulty = 3
    return {
        "course": course,
        "topic": topic,
        "subtopic": subtopic,
        "difficulty": difficulty,
        "prerequisites": list(prerequisites),
    }


def _render_taxonomy_controls(
    index: int,
    database: Database,
    staged: StagedQuestion,
) -> tuple[str, str, str | None, int, list[str], list[str]]:
    """Dibuja la taxonomía y devuelve los valores elegidos.

    Los controles se **pre-rellenan** con la clasificación curricular inferida
    por el LLM en el pase único de estructuración y permanecen editables.
    """
    defaults = _taxonomy_defaults(staged)
    course_default = defaults["course"]
    topic_default = defaults["topic"]
    subtopic_default = defaults["subtopic"] or ""
    difficulty_default = defaults["difficulty"]

    courses = sorted({row["slug"] for row in database.list_courses()})
    course = st.text_input(
        "Curso (slug snake_case)",
        value=course_default,
        key=f"course_{index}",
        help=f"Cursos existentes: {', '.join(courses) or '(ninguno)'}",
    )
    topic = st.text_input(
        "Tema (slug snake_case)",
        value=topic_default,
        key=f"topic_{index}",
        help="Se crea en el catálogo al confirmar si no existe.",
    )
    subtopic_raw = st.text_input(
        "Subtema (slug snake_case, opcional)",
        value=subtopic_default,
        key=f"subtopic_{index}",
    )
    difficulty = st.slider(
        "Dificultad (1-5, complejidad matemática)",
        min_value=1,
        max_value=5,
        value=int(difficulty_default),
        key=f"diff_{index}",
    )

    prereq_default = list(dict.fromkeys(defaults["prerequisites"]))
    catalog = sorted({row["slug"] for row in database.list_prerequisites()})
    options = sorted({*catalog, *prereq_default})
    prerequisites = st.multiselect(
        "Prerrequisitos",
        options,
        default=[slug for slug in prereq_default if slug in options],
        key=f"prereq_{index}",
    )
    new_prereqs_raw = st.text_input(
        "Nuevos prerrequisitos (separados por coma)", key=f"new_prereq_{index}"
    )
    return (
        course,
        topic,
        _slugify(subtopic_raw) or None,
        difficulty,
        prerequisites,
        _slug_split(new_prereqs_raw),
    )


def _sync_question_widgets(index: int, staged: StagedQuestion) -> None:
    """Sobrescribe los controles de la pregunta ``index`` con el reanálisis.

    Tras ``refine_question`` el LLM devuelve, en la misma respuesta JSON, el TeX
    corregido y la taxonomía re-inferida completa (``course``, ``topic``,
    ``subtopic``, ``difficulty`` y ``prerequisites``). Escribimos esos valores en
    ``st.session_state`` para que los inputs del formulario se actualicen al
    instante, sin esperar a una edición manual.
    """
    question = staged.question
    if question is None:
        return
    st.session_state[f"statement_{index}"] = question.statement_raw
    st.session_state[f"qtype_{index}"] = question.question_type
    st.session_state[f"course_{index}"] = question.course
    st.session_state[f"topic_{index}"] = question.topic
    st.session_state[f"subtopic_{index}"] = question.subtopic or ""
    st.session_state[f"diff_{index}"] = int(question.difficulty)
    st.session_state[f"prereq_{index}"] = list(question.prerequisites)
    for key in list(st.session_state.keys()):
        if key.startswith((f"choice_id_{index}_", f"choice_text_{index}_")):
            del st.session_state[key]
    for choice_index, choice in enumerate(question.choices or []):
        st.session_state[f"choice_id_{index}_{choice_index}"] = choice.id
        st.session_state[f"choice_text_{index}_{choice_index}"] = choice.content_raw


def _refine_question(index: int) -> None:
    """Callback ``on_click`` que reanaliza una pregunta y refresca el formulario.

    Se ejecuta antes de que la nueva ejecución instancie los widgets, por lo que
    puede actualizar ``st.session_state`` (incluidos los controles del
    formulario) sin provocar ``StreamlitWidgetAlreadyInstantiatedError``. Los
    errores se guardan en la sesión para mostrarlos en el cuerpo del script.
    """
    result: StagingResult | None = st.session_state.get("staging_result")
    if result is None or not 0 <= index < len(result.questions):
        return
    staged = result.questions[index]
    base = staged.question
    if base is None:
        return

    intermediate = result.intermediate
    if intermediate is None:
        intermediate = TranscriptionIntermediate(
            source_reference=result.source_reference,
            raw_markdown=base.statement_raw,
            pages=[],
            figures=base.figures,
            warnings=[],
        )
    hint = st.session_state.get(f"hint_{index}")
    try:
        refined = st.session_state["_pipeline"].refine_question(
            intermediate, base, hints=hint or None
        )
    except Exception as exc:  # noqa: BLE001 - se informa en el cuerpo del script
        st.session_state[f"_refine_error_{index}"] = str(exc)
        return

    st.session_state.pop(f"_refine_error_{index}", None)
    result.questions[index] = refined
    st.session_state["staging_result"] = result
    _sync_question_widgets(index, refined)
    if refined.question is not None:
        _reset_figures_state(index, refined.question.figures)


def _render_review(
    result: StagingResult,
    settings: Settings,
    database: Database,
    storage: ObjectStorage,
    source_path: str | None,
) -> None:
    summary = result.summary()
    metrics = st.columns(4)
    metrics[0].metric("Estado", summary["status"])
    metrics[1].metric("Total", summary["total"])
    metrics[2].metric("VALID", summary["valid"])
    metrics[3].metric("NEEDS_ATTENTION", summary["needs_attention"])
    if summary["session_id"]:
        st.caption(f"Sesión de staging: `{summary['session_id']}`")
    if result.warnings:
        joined = "\n".join(f"- {warning}" for warning in result.warnings)
        st.warning("Advertencias de OCR:\n" + joined)

    st.caption(
        "La taxonomía, la dificultad y los prerrequisitos se infirieron automáticamente "
        "durante la estructuración; revísalos y edítalos antes de confirmar."
    )

    approved: list[tuple[EnrichedQuestion, list[str]]] = []

    for index, staged in enumerate(result.questions):
        label = f"Pregunta {index + 1} — {staged.status}"
        with st.expander(label, expanded=staged.status == STATUS_NEEDS_ATTENTION):
            if staged.status == STATUS_VALID:
                st.success("Validación superada (VALID). Revísala y apruébala para guardarla.")
            else:
                st.error("NEEDS_ATTENTION: " + "; ".join(staged.reasons))
            if staged.logical is not None and not staged.logical.is_valid:
                st.error(f"Motivo de rechazo semántico: {staged.logical.rejection_reason}")

            if staged.question is None:
                st.info("No hay enunciado estructurado para editar en esta pregunta.")
                continue

            base = staged.question
            left, right = st.columns(2)
            with left:
                new_statement = st.text_area(
                    "Enunciado (Markdown/TeX)", value=base.statement_raw, key=f"statement_{index}"
                )
                question_types = ["multiple_choice", "open_ended", "true_false", "numerical"]
                qtype = st.selectbox(
                    "Tipo de pregunta",
                    question_types,
                    index=question_types.index(base.question_type),
                    key=f"qtype_{index}",
                )
                new_choices: list[Choice] | None = None
                if qtype == "multiple_choice":
                    choices: list[Choice] = []
                    for choice_index, choice in enumerate(base.choices or []):
                        choice_id = st.text_input(
                            f"Alternativa {choice_index + 1} — id",
                            value=choice.id,
                            key=f"choice_id_{index}_{choice_index}",
                        )
                        choice_text = st.text_area(
                            f"Alternativa {choice_index + 1} — contenido",
                            value=choice.content_raw,
                            key=f"choice_text_{index}_{choice_index}",
                        )
                        choices.append(Choice(id=choice_id, content_raw=choice_text))
                    new_choices = choices
                if base.notation_notes:
                    st.info("Notas de notación: " + "; ".join(base.notation_notes))
            with right:
                st.caption("Enunciado renderizado")
                st.markdown(new_statement)
                _render_figure_review(index, staged, source_path, storage, settings)

            st.text_input(
                "Contexto correctivo (p. ej. «La condición ilegible es g(x) = ...»)",
                key=f"hint_{index}",
            )
            refine_error = st.session_state.pop(f"_refine_error_{index}", None)
            if refine_error:
                st.error(f"No se pudo reanalizar: {refine_error}")
            st.button(
                "Reanalizar enunciado",
                key=f"refine_{index}",
                on_click=_refine_question,
                args=(index,),
            )

            course, topic, subtopic, difficulty, prerequisites, new_prereqs = (
                _render_taxonomy_controls(index, database, staged)
            )

            # Reconciliar figuras editadas con el enunciado (correspondencia 1:1).
            edited_figures = _figures_state(index, base.figures)
            efectivas, huerfanas, colgantes = reconcile_figures_with_statement(
                edited_figures, new_statement
            )
            for figure in huerfanas:
                _register_orphan(base.id, figure.get("asset_path"))

            approved_question = build_enriched_question(
                base,
                course=course.strip(),
                topic=topic.strip(),
                subtopic=subtopic,
                difficulty=difficulty,
                prerequisites=[*prerequisites, *new_prereqs],
                source_metadata={"source_reference": result.source_reference},
                figures=[FigureReference.model_validate(figure) for figure in efectivas],
            )
            # Reflejar la edición manual en el objeto que se usará al confirmar.
            # ``model_copy`` no revalida: sanitiza el enunciado manual para no
            # persistir delimitadores TeX no estándar (\[...\], \(...\)).
            approved_question = approved_question.model_copy(
                update={
                    "statement_raw": sanitize_math_delimiters(new_statement),
                    "question_type": qtype,
                    "choices": new_choices,
                }
            )
            if colgantes:
                st.error(
                    "El enunciado referencia placeholders sin figura declarada: "
                    + ", ".join(sorted(colgantes))
                    + ". Elimina el placeholder o restaura la figura para poder aprobar."
                )
            approve = st.checkbox(
                "Aprobar y guardar esta pregunta",
                value=staged.status == STATUS_VALID and not colgantes,
                key=f"approve_{index}",
            )
            if approve and not colgantes:
                approved.append((approved_question, new_prereqs))

    st.divider()
    if st.button("Confirmar y persistir preguntas aprobadas", type="primary"):
        _confirm(approved, result, database, storage)


def _confirm(
    approved: list[tuple[EnrichedQuestion, list[str]]],
    result: StagingResult,
    database: Database,
    storage: ObjectStorage,
) -> None:
    if not approved:
        st.warning("No hay preguntas aprobadas para guardar.")
        return
    invalid = [
        q for q, _ in approved if not _SLUG_RE.match(q.course) or not _SLUG_RE.match(q.topic)
    ]
    if invalid:
        st.error("Curso y tema deben ser slugs válidos (minúsculas, dígitos y guion bajo).")
        return
    new_tags = sorted({slug for _, slugs in approved for slug in slugs})
    try:
        for question, _ in approved:
            database.upsert_course(question.course, format_course_name(question.course))
            database.upsert_topic(question.course, question.topic, question.subtopic)
        if new_tags:
            database.add_prerequisites(new_tags)
    except DatabaseError as exc:
        st.error(f"No se pudo actualizar la taxonomía: {exc}")
        return

    confirm_result = persist_approved_questions(
        [question for question, _ in approved],
        database,
        storage,
        session_id=result.session_id,
    )
    _purge_orphan_assets(set(confirm_result.saved), storage)
    if confirm_result.saved:
        st.success(f"Persistidas {len(confirm_result.saved)} pregunta(s) en taygeta.db.")
    for error in confirm_result.errors:
        st.error(error)


def _sidebar(settings: Settings, database: Database, storage: ObjectStorage) -> None:
    with st.sidebar:
        st.header("Estado")
        try:
            status = database.status()
            st.write(
                {
                    "db": status["db_path"],
                    "integridad": status["integrity"],
                    "fts5": status["fts5_enabled"],
                    "preguntas": status["counts"]["questions"],
                }
            )
        except DatabaseError as exc:
            st.error(f"Base de datos no disponible: {exc}")
        public_endpoint = public_s3_url(settings.S3_ENDPOINT_URL, settings)
        st.caption(f"S3: {settings.S3_BUCKET} @ {public_endpoint}")
        if public_endpoint and public_endpoint != settings.S3_ENDPOINT_URL:
            st.caption(f"(endpoint interno: {settings.S3_ENDPOINT_URL})")
        if not settings.llm_configured:
            st.warning(
                "LLM remoto sin configurar: la validación semántica quedará en NEEDS_ATTENTION."
            )
        with st.expander("Taxonomía"):
            st.write("Cursos:", [c["slug"] for c in database.list_courses()])
            st.write("Prerrequisitos:", [p["slug"] for p in database.list_prerequisites()])


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------


def _render_ingest_tab(
    settings: Settings,
    database: Database,
    storage: ObjectStorage,
    pipeline: IngestPipeline,
) -> None:
    """Pestaña de subida, procesamiento y revisión Human-in-the-Loop."""
    st.subheader("1. Subir documento")
    uploaded = st.file_uploader(
        "PDF, PNG o JPG", type=UPLOAD_EXTENSIONS, accept_multiple_files=False
    )
    if uploaded is not None and st.button("Guardar en uploads"):
        path = save_upload(uploaded.name, uploaded.getvalue(), settings.UPLOADS_DIR)
        _guard_staging_source(str(path))
        st.session_state["processed_path"] = str(path)
        st.success(f"Guardado en {path}")

    uploads = list_uploads(settings.UPLOADS_DIR)
    if not uploads:
        st.info("No hay documentos en el directorio de uploads.")
        return

    st.subheader("2. Procesar")
    selected_name = st.selectbox("Documento a procesar", [p.name for p in uploads])
    selected_path = next(path for path in uploads if path.name == selected_name)
    process_hint = st.text_area("Contexto para el análisis (opcional)", key="process_hint")
    if st.button("Procesar documento", type="primary"):
        with st.status("Ejecutando ingesta híbrida…", expanded=True) as status:
            st.write("Rasterizando y ejecutando el OCR local…")
            st.write("Estructurando con el LLM remoto y validando…")
            try:
                result = pipeline.ingest(selected_path, hints=process_hint or None)
            except Exception as exc:  # noqa: BLE001
                status.update(label="La ingesta falló", state="error")
                st.error(f"Error de ingesta: {exc}")
                return
            _guard_staging_source(str(selected_path))
            st.session_state["staging_result"] = result
            st.session_state["processed_path"] = str(selected_path)
            status.update(label=f"Ingesta completada: {result.status}", state="complete")

    result: StagingResult | None = st.session_state.get("staging_result")
    if result is None:
        return

    st.subheader("3. Revisión y confirmación")
    _render_review(result, settings, database, storage, st.session_state.get("processed_path"))


def _render_question_bank(database: Database, settings: Settings) -> None:
    """Pestaña de exploración del banco de preguntas guardadas (SQLite + FTS5)."""
    st.subheader("Banco de preguntas")
    try:
        status = database.status()
    except DatabaseError as exc:
        st.error(f"Base de datos no disponible: {exc}")
        return
    st.metric("Preguntas registradas", status["counts"]["questions"])

    courses = ["(todos)", *sorted({c["slug"] for c in database.list_courses()})]
    course = st.selectbox("Filtrar por curso", courses, key="bank_course")
    course_filter = None if course == "(todos)" else course

    topics = ["(todos)"]
    if course_filter:
        topics += sorted({t["topic_slug"] for t in database.list_topics(course_filter)})
    topic = st.selectbox("Filtrar por tema", topics, key="bank_topic")
    topic_filter = None if topic == "(todos)" else topic

    query = st.text_input("Buscar en el enunciado (Markdown/TeX, FTS5)", key="bank_query").strip()
    limit = st.slider("Máximo de resultados", 5, 100, 20, step=5, key="bank_limit")
    try:
        questions = database.search_questions(
            query=query or None,
            course=course_filter,
            topic=topic_filter,
            limit=limit,
        )
    except DatabaseError as exc:
        st.error(f"Error en la búsqueda: {exc}")
        return

    st.caption(f"{len(questions)} resultado(s).")
    if not questions:
        st.info("No hay preguntas que cumplan los filtros.")
        return
    for question in questions:
        header = f"[{question.course}/{question.topic}] dificultad {question.difficulty}"
        with st.expander(header):
            st.markdown(question.statement_raw)
            st.json(
                {
                    "id": question.id,
                    "course": question.course,
                    "topic": question.topic,
                    "subtopic": question.subtopic,
                    "difficulty": question.difficulty,
                    "prerequisites": question.prerequisites,
                    "created_at": question.created_at,
                }
            )


def main() -> None:
    """Renderiza la aplicación Streamlit (ingesta/revisión + banco de preguntas)."""
    st.set_page_config(page_title="Taygeta — Revisión HITL", layout="wide")
    st.title("Taygeta — Ingesta y revisión Human-in-the-Loop")

    try:
        settings, database, storage, pipeline = _resources()
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudieron inicializar los servicios: {exc}")
        return
    st.session_state["_pipeline"] = pipeline
    _sync_url_session()
    _sidebar(settings, database, storage)

    tab_ingest, tab_bank = st.tabs(["Ingesta y revisión", "Banco de preguntas"])
    with tab_ingest:
        _render_ingest_tab(settings, database, storage, pipeline)
    with tab_bank:
        _render_question_bank(database, settings)


if __name__ == "__main__":
    main()
