"""Taxonomía curricular de Taygeta: nombres legibles de curso.

Este módulo conserva únicamente los helpers de presentación del dominio
curricular:

* :data:`COURSE_NAMES`: catálogo de nombres legibles de curso (con tildes y
  mayúsculas de estilo título) asociados a sus slugs.
* :func:`title_case_spanish` y :func:`format_course_name`.

La canonicalización de ``course``, ``topic``, ``subtopic`` y prerrequisitos es
100 % dinámica: se lee de la base de datos en vivo (ver
:meth:`taygeta.server.db.Database.get_live_taxonomy_tree` y
:meth:`taygeta.server.db.Database.list_prerequisites`) y se inyecta en los system
prompts de estructuración. No hay diccionarios de alias ni vocabulario cerrado
hardcodeado.

Ninguno de estos helpers almacena soluciones ni respuestas correctas: son
únicamente nombres de presentación.
"""

from __future__ import annotations

__all__ = [
    "COURSE_NAMES",
    "format_course_name",
    "title_case_spanish",
]

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
