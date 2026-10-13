"""Pruebas de la edición manual de figuras en la revisión HITL.

Cubre los helpers puros de ``taygeta.ui.figure_editing`` (conversión de
píxeles/BBox, limpieza de placeholders y reconciliación), el filtro de
página completa de ``detect_figures`` y las nuevas restricciones de los
prompts (no declarar ``fig:N`` sin un diagrama explícito).
"""

from __future__ import annotations

import pymupdf as fitz
import pytest

from taygeta.agents.structurer import _REFINING_SYSTEM_PROMPT, _STRUCTURING_SYSTEM_PROMPT
from taygeta.ingest.image_extractor import detect_figures, placeholder_for
from taygeta.ingest.pipeline import IngestPipeline
from taygeta.ui.figure_editing import (
    append_figure_placeholder,
    bbox_to_pixel_box,
    build_new_figure,
    next_figure_id,
    pixel_box_to_normalized_bbox,
    reconcile_figures_with_statement,
    remove_figure,
    strip_figure_placeholder,
)


class _FakePage:
    """Página PyMuPDF mínima para probar ``detect_figures`` sin PDF real."""

    def __init__(
        self,
        images: list[tuple[int, list[fitz.Rect]]] | None = None,
        drawings: list[fitz.Rect] | None = None,
        width: float = 612.0,
        height: float = 792.0,
    ) -> None:
        self.rect = fitz.Rect(0, 0, width, height)
        self.number = 0
        self._images = images or []
        self._drawings = drawings or []

    def get_images(self, full: bool = True) -> list[tuple[int, int]]:
        return [(xref, 0) for xref, _rects in self._images]

    def get_image_rects(self, xref: int) -> list[fitz.Rect]:
        for current_xref, rects in self._images:
            if current_xref == xref:
                return list(rects)
        return []

    def get_drawings(self) -> list[dict[str, fitz.Rect]]:
        return [{"rect": rect} for rect in self._drawings]


# ---------------------------------------------------------------------------
# Conversión píxeles <-> BBox normalizada
# ---------------------------------------------------------------------------


def test_pixel_box_to_normalized_bbox() -> None:
    box = {"left": 100, "top": 200, "width": 200, "height": 200}
    assert pixel_box_to_normalized_bbox(box, width=1000, height=800) == [250, 100, 500, 300]


def test_bbox_to_pixel_box_roundtrip() -> None:
    bbox = [250, 100, 500, 300]
    xl, xr, yt, yb = bbox_to_pixel_box(bbox, width=1000, height=800)
    assert (xl, xr, yt, yb) == (100, 300, 200, 400)


@pytest.mark.parametrize(
    ("box", "width", "height"),
    [
        ({"left": 0, "top": 0, "width": 0, "height": 50}, 1000, 800),
        ({"left": 0, "top": 0, "width": 50, "height": 0}, 1000, 800),
        ({"left": -5, "top": 0, "width": 50, "height": 50}, 1000, 800),
        ({"left": 0, "top": 0, "width": 1100, "height": 50}, 1000, 800),
    ],
)
def test_pixel_box_invalid_rejected(box: dict[str, int], width: int, height: int) -> None:
    with pytest.raises(ValueError):
        pixel_box_to_normalized_bbox(box, width, height)


# ---------------------------------------------------------------------------
# Limpieza de placeholders y reconciliación
# ---------------------------------------------------------------------------


def test_strip_figure_placeholder() -> None:
    statement = "Según la figura {{fig:1}} y {{fig:1}} resuelve:"
    assert strip_figure_placeholder(statement, "fig:1") == "Según la figura  y  resuelve:"


def test_strip_figure_placeholder_keeps_others() -> None:
    statement = "Usa {{fig:1}} junto con {{fig:2}}."
    assert strip_figure_placeholder(statement, "fig:1") == "Usa  junto con {{fig:2}}."


def test_reconcile_figures_with_statement() -> None:
    figures = [
        {"id": "fig:1", "page": 0, "bbox": [100, 100, 400, 400]},
        {"id": "fig:2", "page": 0, "bbox": [500, 100, 700, 300]},
    ]
    efectivas, huerfanas, colgantes = reconcile_figures_with_statement(
        figures, "Observa {{fig:1}} y responde."
    )
    assert [figure["id"] for figure in efectivas] == ["fig:1"]
    assert [figure["id"] for figure in huerfanas] == ["fig:2"]
    assert colgantes == []


def test_reconcile_detects_orphan_placeholders() -> None:
    figures = [{"id": "fig:1", "page": 0, "bbox": [100, 100, 400, 400]}]
    efectivas, huerfanas, colgantes = reconcile_figures_with_statement(
        figures, "Analiza {{fig:1}} y {{fig:2}}."
    )
    assert len(efectivas) == 1
    assert huerfanas == []
    assert colgantes == ["fig:2"]


# ---------------------------------------------------------------------------
# Añadir figuras manualmente (falsos negativos): id secuencial y placeholder
# ---------------------------------------------------------------------------


def test_next_figure_id_starts_at_one() -> None:
    assert next_figure_id([], "") == "fig:1"


def test_next_figure_id_sequential_with_gaps() -> None:
    figures = [
        {"id": "fig:1", "page": 0, "bbox": [0, 0, 1, 1]},
        {"id": "fig:3", "page": 0, "bbox": [0, 0, 1, 1]},
    ]
    assert next_figure_id(figures, "Observa {{fig:1}}.") == "fig:4"


def test_next_figure_id_avoids_dangling_placeholders() -> None:
    figures = [{"id": "fig:1", "page": 0, "bbox": [0, 0, 1, 1]}]
    statement = "Analiza {{fig:1}} y {{fig:2}}."
    assert next_figure_id(figures, statement) == "fig:3"


def test_next_figure_id_ignores_non_numeric_ids() -> None:
    assert next_figure_id([{"id": "fig:abc", "page": 0, "bbox": [0, 0, 1, 1]}], "texto") == "fig:1"


def test_append_figure_placeholder() -> None:
    assert append_figure_placeholder("Resuelve.", "fig:2") == "Resuelve.\n{{fig:2}}"
    assert append_figure_placeholder("", "fig:1") == "{{fig:1}}"
    assert append_figure_placeholder("   ", "fig:1") == "{{fig:1}}"


# ---------------------------------------------------------------------------
# Alta y baja de figuras (helpers puros build_new_figure / remove_figure)
# ---------------------------------------------------------------------------


def test_build_new_figure_asigna_siguiente_id_y_placeholder() -> None:
    statement = "Analiza {{fig:1}} y responde."
    figures = [{"id": "fig:1", "page": 0, "bbox": [0, 0, 1, 1], "asset_path": "a.png"}]
    nuevas, nuevo_enunciado, figure_id = build_new_figure(
        figures,
        statement,
        page_index=2,
        bbox=[100, 100, 400, 300],
        asset_path="/buckets/taygeta-assets/figures/nueva.png",
    )
    assert figure_id == "fig:2"
    assert nuevo_enunciado == "Analiza {{fig:1}} y responde.\n{{fig:2}}"
    assert [fig["id"] for fig in nuevas] == ["fig:1", "fig:2"]
    assert nuevas[-1] == {
        "id": "fig:2",
        "page": 2,
        "bbox": [100, 100, 400, 300],
        "caption": None,
        "asset_path": "/buckets/taygeta-assets/figures/nueva.png",
    }
    # El helper es puro: la lista original no se muta.
    assert len(figures) == 1


def test_build_new_figure_con_lista_vacia() -> None:
    nuevas, nuevo_enunciado, figure_id = build_new_figure(
        [],
        "",
        page_index=0,
        bbox=[1, 2, 3, 4],
        asset_path="k.png",
    )
    assert figure_id == "fig:1"
    assert nuevo_enunciado == "{{fig:1}}"
    assert len(nuevas) == 1


def test_remove_figure_quita_figura_y_placeholder() -> None:
    figures = [
        {"id": "fig:1", "page": 0, "bbox": [0, 0, 1, 1], "asset_path": "a.png"},
        {"id": "fig:2", "page": 1, "bbox": [0, 0, 1, 1], "asset_path": "b.png"},
    ]
    statement = "Observa {{fig:1}} y luego {{fig:2}}."
    kept, limpio, removed = remove_figure(figures, statement, "fig:1")
    assert [fig["id"] for fig in kept] == ["fig:2"]
    assert "{{fig:1}}" not in limpio
    assert "{{fig:2}}" in limpio
    assert removed == [{"id": "fig:1", "page": 0, "bbox": [0, 0, 1, 1], "asset_path": "a.png"}]
    # La lista original no se muta.
    assert len(figures) == 2


def test_remove_figure_inexistente_no_cambia_nada() -> None:
    figures = [{"id": "fig:1", "page": 0, "bbox": [0, 0, 1, 1], "asset_path": "a.png"}]
    kept, limpio, removed = remove_figure(figures, "Con {{fig:1}}.", "fig:9")
    assert kept == figures
    assert limpio == "Con {{fig:1}}."
    assert removed == []


# ---------------------------------------------------------------------------
# Detección: descarte de página completa
# ---------------------------------------------------------------------------


def test_detect_figures_drops_full_page_scan() -> None:
    page = _FakePage(images=[(1, [fitz.Rect(0, 0, 612, 792)])])
    assert detect_figures(page) == []


def test_detect_figures_drops_near_full_page_and_keeps_small() -> None:
    full = fitz.Rect(0, 0, 500, 700)  # ≈ 72.2 % del área -> se descarta
    small = fitz.Rect(520, 50, 560, 150)  # ≈ 0.8 % -> se conserva
    page = _FakePage(images=[(1, [full]), (2, [small])])
    figures = detect_figures(page)
    assert [figure.id for figure in figures] == ["fig:1"]
    assert figures[0].bbox[0] == round(50 / 792 * 1000)  # ymin del rect pequeño


def test_detect_figures_keeps_regular_figure() -> None:
    page = _FakePage(images=[(1, [fitz.Rect(100, 100, 400, 300)])])
    figures = detect_figures(page)
    assert len(figures) == 1
    assert placeholder_for(figures[0].id) == "{{fig:1}}"


def test_detect_figures_ignores_rule_drawings() -> None:
    page = _FakePage(drawings=[fitz.Rect(0, 10, 612, 11)])  # regla horizontal
    assert detect_figures(page) == []


# ---------------------------------------------------------------------------
# Prompts: solo figuras explícitas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prompt", [_STRUCTURING_SYSTEM_PROMPT, _REFINING_SYSTEM_PROMPT])
def test_prompts_require_explicit_figures(prompt: str) -> None:
    assert "NO declares fig:N" in prompt


def test_figure_hint_no_longer_orders_placeholders() -> None:
    from taygeta.core.schema import FigureReference

    hint = IngestPipeline._figure_hint(
        [FigureReference(id="fig:1", page=0, bbox=[100, 100, 500, 500])], page_index=0
    )
    assert "SOLO si el enunciado depende" in hint
    assert "insertes ningún placeholder" in hint
