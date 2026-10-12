"""Taxonomía curricular canónica de Taygeta.

Define la fuente de verdad del dominio curricular:

* :data:`PREREQUISITE_ALIASES`: diccionario de *aliasing* de prerrequisitos que
  mapea variantes de slug hacia su slug canónico único. Se usa para
  canonicalizar y deduplicar los prerrequisitos de cada pregunta.
* :data:`CANONICAL_PREREQUISITES`: lista ordenada y estable de los slugs
  canónicos derivados de :data:`PREREQUISITE_ALIASES`. Se inyecta en los system
  prompts de estructuración para orientar al LLM hacia el vocabulario cerrado.
* :data:`COURSE_NAMES`: catálogo de nombres legibles de curso (con tildes y
  mayúsculas de estilo título) asociados a sus slugs.
* Helpers :func:`canonical_prerequisite`, :func:`canonicalize_prerequisites`,
  :func:`title_case_spanish` y :func:`format_course_name`.

Ninguno de estos diccionarios almacena soluciones ni respuestas correctas: son
únicamente vocabulario curricular.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = [
    "CANONICAL_PREREQUISITES",
    "COURSE_NAMES",
    "PREREQUISITE_ALIASES",
    "canonical_prerequisite",
    "canonicalize_prerequisites",
    "format_course_name",
    "title_case_spanish",
]

#: Alias de prerrequisito → slug canónico. Las variantes que emiten los modelos
#: OCR/LLM (singular/plural, sinónimos) se resuelven a un único slug estable.
PREREQUISITE_ALIASES: dict[str, str] = {
    "sucesion_de_cauchy": "sucesiones_de_cauchy",
    "espacios_vectoriales": "espacio_vectorial",
    "subespacio_vectorial": "subespacios_vectoriales",
    "metricas": "metrica",
    "espacio_vectorial_normado": "espacios_normados",
    "espacios_vectoriales_normados": "espacios_normados",
    "espacio_de_sucesiones_acotadas": "sucesiones_acotadas",
    "derivadas": "derivada",
    "funciones_continuas": "continuidad",
}

#: Slugs canónicos únicos (valores de :data:`PREREQUISITE_ALIASES`) en orden
#: estable de primera aparición.
CANONICAL_PREREQUISITES: tuple[str, ...] = tuple(dict.fromkeys(PREREQUISITE_ALIASES.values()))

#: Palabras menores que, en un Title Case adaptado al español, se escriben en
#: minúscula salvo que encabecen el título.
_SPANISH_TITLE_CASE_MINOR = frozenset(
    {
        "de",
        "del",
        "el",
        "la",
        "las",
        "los",
        "y",
        "e",
        "o",
        "u",
        "a",
        "en",
        "con",
        "para",
        "por",
        "sin",
        "sobre",
        "entre",
    }
)

#: Nombres legibles de curso (con tildes y mayúsculas de estilo título).
#: Catálogo de partida, extensible según el dominio; cualquier slug no listado
#: se resuelve mediante :func:`title_case_spanish`.
COURSE_NAMES: dict[str, str] = {
    "analisis_funcional": "Análisis Funcional",
    "analisis_real": "Análisis Real",
    "analisis_complejo": "Análisis Complejo",
    "calculo": "Cálculo",
    "calculo_diferencial": "Cálculo Diferencial",
    "calculo_integral": "Cálculo Integral",
    "ecuaciones_diferenciales": "Ecuaciones Diferenciales",
    "algebra": "Álgebra",
    "algebra_lineal": "Álgebra Lineal",
    "algebra_abstracta": "Álgebra Abstracta",
    "topologia": "Topología",
    "geometria": "Geometría",
    "probabilidad": "Probabilidad",
    "estadistica": "Estadística",
    "variable_compleja": "Variable Compleja",
}


def canonical_prerequisite(slug: str) -> str:
    """Devuelve el slug canónico de un prerrequisito, o el propio si no tiene alias."""
    return PREREQUISITE_ALIASES.get(slug, slug)


def canonicalize_prerequisites(slugs: Iterable[str]) -> list[str]:
    """Canonicaliza y deduplica una lista de prerrequisitos preservando el orden.

    Cada slug se resuelve contra :data:`PREREQUISITE_ALIASES`; los duplicados
    resultantes y los vacíos se descartan.
    """
    result: list[str] = []
    for slug in slugs:
        canonical = PREREQUISITE_ALIASES.get(slug, slug)
        if canonical and canonical not in result:
            result.append(canonical)
    return result


def title_case_spanish(slug: str) -> str:
    """Convierte un slug ``snake_case`` en un nombre en Title Case adaptado al español.

    Capitaliza cada palabra salvo las menores (artículos, preposiciones y
    conjunciones habituales), que se mantienen en minúscula cuando no encabezan
    el título.
    """
    words = slug.replace("_", " ").split()
    if not words:
        return slug
    result: list[str] = []
    for index, word in enumerate(words):
        lowered = word.lower()
        if index != 0 and lowered in _SPANISH_TITLE_CASE_MINOR:
            result.append(lowered)
        else:
            result.append(lowered.capitalize())
    return " ".join(result)


def format_course_name(slug: str) -> str:
    """Devuelve el nombre legible de un curso (con tildes) o un Title Case adaptado.

    Consulta primero :data:`COURSE_NAMES` y, si el slug no está catalogado, aplica
    :func:`title_case_spanish` como respaldo.
    """
    known = COURSE_NAMES.get(slug)
    if known:
        return known
    return title_case_spanish(slug)
