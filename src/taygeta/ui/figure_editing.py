"""Edición manual de figuras en la revisión HITL (helpers puros, sin Streamlit).

Centraliza la lógica reutilizable entre la UI y los tests:

* conversión del recuadro de píxeles del cropper a la BBox normalizada
  ``[ymin, xmin, ymax, xmax]`` (0-1000) que exige ``FigureReference``;
* conversión inversa (BBox normalizada -> píxeles) para sembrar el recuadro
  inicial del cropper;
* eliminación del placeholder ``{{fig:N}}`` del enunciado al borrar una figura;
* reconciliación figuras/placeholders conservando la correspondencia uno a uno
  que impone ``QuestionRaw`` al confirmar la pregunta.

Este módulo no importa Streamlit para que las funciones sean unit-testables.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from taygeta.core.schema import PLACEHOLDER_PATTERN

__all__ = [
    "NORMALIZED_MAX",
    "bbox_to_pixel_box",
    "pixel_box_to_normalized_bbox",
    "reconcile_figures_with_statement",
    "strip_figure_placeholder",
]

#: Escala de la BBox normalizada (misma constante que ``image_extractor``).
NORMALIZED_MAX = 1000

_PLACEHOLDER_RE = re.compile(PLACEHOLDER_PATTERN)


def _figure_id(figure: Mapping[str, Any] | Any) -> str:
    """Devuelve el ``id`` de una figura representada como modelo o mapping."""
    if isinstance(figure, Mapping):
        return str(figure.get("id") or "")
    return str(getattr(figure, "id", "") or "")


def pixel_box_to_normalized_bbox(
    box: Mapping[str, int],
    width: int,
    height: int,
) -> list[int]:
    """Convierte el recuadro del cropper (píxeles) en BBox normalizada 0-1000.

    ``box`` usa las claves ``left``/``top``/``width``/``height`` que devuelve
    ``st_cropper(return_type="box")`` sobre una imagen de ``width`` x ``height``
    píxeles. Devuelve ``[ymin, xmin, ymax, xmax]`` con las mismas garantías que
    el validador de ``FigureReference`` (coordenadas entre 0 y 1000 y lados no
    degenerados).

    Raises:
        ValueError: si las dimensiones no son positivas o el recuadro queda
            fuera de los límites o degenerado tras normalizar.
    """
    if width <= 0 or height <= 0:
        raise ValueError("las dimensiones de la imagen deben ser positivas")

    left = int(box.get("left", 0))
    top = int(box.get("top", 0))
    box_width = int(box.get("width", 0))
    box_height = int(box.get("height", 0))
    if box_width <= 0 or box_height <= 0:
        raise ValueError("el recuadro de recorte debe tener ancho y alto positivos")
    if left < 0 or top < 0 or left + box_width > width or top + box_height > height:
        raise ValueError("el recuadro de recorte queda fuera de los límites de la imagen")

    xmin = round(left / width * NORMALIZED_MAX)
    ymin = round(top / height * NORMALIZED_MAX)
    xmax = round((left + box_width) / width * NORMALIZED_MAX)
    ymax = round((top + box_height) / height * NORMALIZED_MAX)

    if ymin >= ymax or xmin >= xmax:
        raise ValueError("el recuadro de recorte queda degenerado tras normalizar")

    return [
        min(max(ymin, 0), NORMALIZED_MAX),
        min(max(xmin, 0), NORMALIZED_MAX),
        min(max(ymax, 0), NORMALIZED_MAX),
        min(max(xmax, 0), NORMALIZED_MAX),
    ]


def bbox_to_pixel_box(
    normalized_bbox: Sequence[int | float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Convierte una BBox normalizada en píxeles: ``(xl, xr, yt, yb)``.

    El formato ``(xl, xr, yt, yb)`` es el que espera ``st_cropper`` en
    ``default_coords`` para sembrar el recuadro inicial del recorte.

    Raises:
        ValueError: si las dimensiones no son positivas o la BBox no tiene 4
            coordenadas.
    """
    if width <= 0 or height <= 0:
        raise ValueError("las dimensiones de la imagen deben ser positivas")
    if len(normalized_bbox) != 4:
        raise ValueError("la BBox normalizada debe tener cuatro coordenadas")
    ymin, xmin, ymax, xmax = (float(value) for value in normalized_bbox)
    xl = round(xmin / NORMALIZED_MAX * width)
    xr = round(xmax / NORMALIZED_MAX * width)
    yt = round(ymin / NORMALIZED_MAX * height)
    yb = round(ymax / NORMALIZED_MAX * height)
    return xl, max(xr, xl + 1), yt, max(yb, yt + 1)


def strip_figure_placeholder(statement: str, figure_id: str) -> str:
    """Elimina todos los placeholders ``{{fig:N}}`` de ``figure_id`` del enunciado.

    Devuelve el enunciado sin esos placeholders (p. ej. ``{{fig:1}}`` se
    elimina al borrar la figura ``fig:1``). Si no aparece, se devuelve intacto.
    """
    pattern = re.compile(r"\{\{" + re.escape(figure_id) + r"\}\}")
    return pattern.sub("", statement)


def reconcile_figures_with_statement(
    figures: Sequence[Mapping[str, Any] | Any],
    statement: str,
) -> tuple[list[Any], list[Any], list[str]]:
    """Reconcilia las figuras con los placeholders del enunciado editado.

    Devuelve una tupla ``(efectivas, huérfanas, colgantes)``:

    * ``efectivas``: figuras cuyo ``id`` aparece como placeholder en el
      enunciado (cuentan en la pregunta final);
    * ``huérfanas``: figuras declaradas que ya no se referencian (conviene
      borrar su asset del bucket);
    * ``colgantes``: identificadores ``fig:N`` presentes en el enunciado sin
      figura declarada (invalidarían ``QuestionRaw`` y deben bloquear la
      aprobación).

    Respeto el orden original de ``figures`` en ``efectivas``; la renumeración
    posterior, si se desea, es decisión del flujo de revisión.
    """
    referenced: set[str] = {f"fig:{match}" for match in _PLACEHOLDER_RE.findall(statement)}
    declared: list[Any] = list(figures)

    colgantes = sorted(referenced - {_figure_id(figure) for figure in declared})
    efectivas = [figure for figure in declared if _figure_id(figure) in referenced]
    huerfanas = [figure for figure in declared if _figure_id(figure) not in referenced]
    return efectivas, huerfanas, colgantes
