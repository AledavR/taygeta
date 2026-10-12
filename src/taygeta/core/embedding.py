"""Utilidades numéricas para la deduplicación semántica por embeddings.

Taygeta calcula la similitud coseno en memoria (Python puro, sin dependencias
extra) porque el volumen de preguntas es bajo. Los vectores se almacenan en
SQLite como texto JSON en la columna ``questions.embedding_json``.

Ninguna de estas funciones maneja soluciones ni respuestas correctas: operan
únicamente sobre vectores numéricos derivados del contenido visible del
enunciado y sus alternativas.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

__all__ = ["cosine_similarity", "decode_embedding", "encode_embedding"]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Devuelve la similitud coseno entre dos vectores, en el rango [-1, 1].

    Si alguno de los vectores está vacío, difieren en longitud o tienen norma
    cero, devuelve ``0.0`` (sin similitud) en lugar de elevar un error.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def encode_embedding(vector: Sequence[float] | None) -> str | None:
    """Serializa un vector de embedding como texto JSON, o ``None`` si está vacío."""
    if not vector:
        return None
    return json.dumps([float(x) for x in vector])


def decode_embedding(value: str | None) -> list[float]:
    """Deserializa un embedding JSON almacenado; devuelve ``[]`` si es inválido."""
    if not value:
        return []
    try:
        data = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return [float(x) for x in data if isinstance(x, (int, float))]
