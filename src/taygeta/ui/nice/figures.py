"""Operaciones y componentes de figuras para la revisión NiceGUI.

El ``DragBoxCropper`` dibuja el recuadro de recorte con arrastre sobre
``ui.interactive_image`` (overlay SVG en coordenadas de píxeles de la imagen,
sin dependencias npm) y entrega el recuadro en píxeles de la imagen original;
las funciones ``crop_new_figure`` y ``replace_figure_crop`` recortan, suben y
registran figuras reutilizando los helpers puros de ``figure_editing`` y los
clientes compartidos de rasterización/almacenamiento.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from nicegui import ui
from PIL import Image

from taygeta.core.urls import split_asset_reference
from taygeta.ingest.image_extractor import DEFAULT_DPI, crop_figure_bytes
from taygeta.server.object_storage import ObjectStorage
from taygeta.ui.figure_editing import build_new_figure
from taygeta.ui.nice import caption as _caption
from taygeta.ui.shared import figure_bytes, page_image_raster


def figure_pil(storage: ObjectStorage | None, asset_path: str | None) -> Image.Image | None:
    """Abre una figura de SeaweedFS como ``PIL.Image`` para ``ui.image``."""
    data = figure_bytes(storage, asset_path)
    if not data:
        return None
    try:
        return Image.open(io.BytesIO(data))
    except OSError:
        return None


def page_pil(
    source_path: str | Path | None, page_index: int
) -> tuple[Image.Image, int, int] | None:
    """Rasteriza la página ``page_index`` como imagen PIL + tamaño en píxeles.

    Devuelve ``(imagen, width_px, height_px)`` o ``None`` si el origen no está
    disponible o la página no existe.
    """
    if source_path is None:
        return None
    raster = page_image_raster(source_path, int(page_index))
    if raster is None:
        return None
    image = Image.open(io.BytesIO(raster.png_bytes))
    image.load()
    return image, int(raster.width_px), int(raster.height_px)


def crop_new_figure(
    figures: Sequence[Mapping[str, Any] | Any],
    statement: str,
    *,
    page_index: int,
    bbox: Sequence[int],
    source_path: str | Path | None,
    storage: ObjectStorage | None,
) -> tuple[list[dict[str, Any]], str, str]:
    """Recorta, sube y registra una figura nueva; devuelve (figuras, enunciado, id).

    Cubre los falsos negativos: recorta la región indicada de la página original
    (300 DPI), la persiste con la convención del pipeline
    (``figures/{uuid}.png``) y delega en ``build_new_figure`` la asignación del
    siguiente ``fig:N`` libre y la inserción de su placeholder al final del
    enunciado.

    Raises:
        ValueError: si no hay origen o almacenamiento disponibles.
    """
    if source_path is None:
        raise ValueError("no hay documento de origen disponible para recortar")
    if storage is None:
        raise ValueError("almacenamiento no disponible para guardar la figura")
    png = crop_figure_bytes(source_path, int(page_index), list(bbox), dpi=DEFAULT_DPI)
    uri = storage.store_asset_bytes(
        png,
        prefix="figures",
        asset_id=None,
        extension=".png",
        content_type="image/png",
    )
    asset_path = storage.relative_path(uri)
    return build_new_figure(
        figures,
        statement,
        page_index=int(page_index),
        bbox=list(bbox),
        asset_path=asset_path,
    )


def replace_figure_crop(
    figures: Sequence[Mapping[str, Any] | Any],
    figure_id: str,
    new_bbox: Sequence[int],
    source_path: str | Path | None,
    storage: ObjectStorage | None,
) -> list[dict[str, Any]]:
    """Re-recorta una figura y sobrescribe su asset (misma clave) en SeaweedFS.

    Recorta la región indicada del documento origen a 300 DPI y sobrescribe el
    objeto existente (misma clave), de modo que ``asset_path`` no cambia.
    Devuelve la lista actualizada con la nueva BBox.

    Raises:
        ValueError: si la figura no tiene asset persistente o faltan servicios.
    """
    if source_path is None:
        raise ValueError("no hay documento de origen disponible para recortar")
    if storage is None:
        raise ValueError("almacenamiento no disponible para guardar el recorte")
    updated: list[dict[str, Any]] = []
    for item in figures:
        if isinstance(item, Mapping):
            figure = dict(item)
        else:
            figure = item.model_dump(mode="json")  # type: ignore[union-attr]
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
            source_path, int(figure.get("page", 0)), list(new_bbox), dpi=DEFAULT_DPI
        )
        storage.put_bytes(key, png, content_type="image/png", bucket=bucket or None)
        updated.append({**figure, "bbox": list(new_bbox)})
    return updated


class DragBoxCropper:
    """Recuadro de recorte dibujado con arrastre sobre una imagen interactiva.

    Captura ``mousedown``/``mousemove``/``mouseup`` sobre ``ui.interactive_image``
    y dibuja el rectángulo como overlay SVG (propiedad ``content``, cuyas
    coordenadas coinciden con los píxeles de la imagen). Al soltar el botón,
    invoca ``on_confirm`` con ``{"left", "top", "width", "height"}`` en píxeles
    de la imagen original (clamped a sus dimensiones).
    """

    def __init__(
        self,
        source: Any,
        *,
        width: int,
        height: int,
        on_confirm: Callable[[dict[str, int]], None],
        caption: str = "Arrastra para seleccionar la región",
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.on_confirm = on_confirm
        self.start: tuple[int, int] | None = None
        self.current: tuple[int, int] | None = None
        _caption(caption)
        self.image = ui.interactive_image(
            source,
            on_mouse=self._on_mouse,
            events=["mousedown", "mousemove", "mouseup"],
            cross=True,
        ).classes("w-full border border-gray-300 rounded")
        self.image.content = ""
        self.image.update()

    def _clamped(self, x: float, y: float) -> tuple[int, int]:
        return max(0, min(int(x), self.width)), max(0, min(int(y), self.height))

    def _on_mouse(self, e: Any) -> None:
        if e.type == "mousedown":
            self.start = self._clamped(e.image_x, e.image_y)
            self.current = self.start
        elif e.type == "mousemove" and self.start is not None:
            self.current = self._clamped(e.image_x, e.image_y)
        elif e.type == "mouseup" and self.start is not None:
            box = self._box()
            self.start = None
            self.current = None
            if box is not None:
                self.on_confirm(box)
        self._refresh()

    def _box(self) -> dict[str, int] | None:
        if self.start is None or self.current is None:
            return None
        x0, y0 = self.start
        x1, y1 = self.current
        left, top = min(x0, x1), min(y0, y1)
        width, height = abs(x1 - x0), abs(y1 - y0)
        if width < 2 or height < 2:
            return None
        return {"left": left, "top": top, "width": width, "height": height}

    def _refresh(self) -> None:
        """Redibuja el overlay SVG con el recuadro actual (si hay arrastre)."""
        box = self._box()
        svg = ""
        if box is not None:
            svg = (
                f'<rect x="{box["left"]}" y="{box["top"]}" width="{box["width"]}" '
                f'height="{box["height"]}" fill="rgba(59,130,246,0.25)" '
                'stroke="#3b82f6" stroke-width="2"/>'
            )
        self.image.content = svg
        self.image.update()
