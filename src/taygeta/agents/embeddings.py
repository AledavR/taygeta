"""Generación 100 % local de embeddings mediante ``sentence-transformers``.

Los embeddings para la deduplicación semántica se calculan con un modelo
``sentence-transformers`` ejecutado localmente en Python (sin llamadas a un
servicio externo ni a una API de embeddings en la nube). El identificador del
modelo se configura con ``LLM_EMBEDDING_MODEL`` (un id de Hugging Face o una
ruta local) y sus pesos se descargan/cachean la primera vez que se usan, igual
que ocurre con los pesos del OCR.

El texto que se vectoriza es el contenido visible del enunciado y sus
alternativas; nunca incluye soluciones ni respuestas correctas (el esquema no
las contiene). Los vectores se normalizan (``normalize_embeddings=True``) para
que la similitud coseno sea directamente el producto escalar.

Si el modelo no puede cargarse (sin pesos, sin red la primera vez, etc.),
:class:`Embedder` eleva :class:`EmbeddingError` y los llamadores deben degradar
de forma controlada: la deduplicación semántica se desactiva sin bloquear la
ingesta.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from taygeta.core.settings import Settings, get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

__all__ = ["Embedder", "EmbeddingError", "question_embedding_text"]


class EmbeddingError(RuntimeError):
    """Error controlado al cargar el modelo local o generar embeddings."""


def question_embedding_text(question: Any) -> str:
    """Compone el texto representativo de una pregunta para vectorizarla.

    Incluye el enunciado y, si las hay, las alternativas. No incorpora solución
    ni respuesta correcta: solo el contenido visible de la pregunta.
    """
    parts = [question.statement_raw]
    for choice in question.choices or []:
        parts.append(choice.content_raw)
    return "\n".join(parts)


class Embedder:
    """Generador local de embeddings con ``sentence-transformers``.

    El modelo se instancia de forma perezosa en :meth:`model` para no descargar
    pesos ni importar ``torch``/``sentence_transformers`` hasta que realmente se
    necesite un vector.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model_name: str | None = None,
        model: Any | None = None,
    ) -> None:
        """Crea el generador de embeddings.

        Args:
            settings: Configuración a usar; por defecto la global.
            model_name: Sobrescribe ``LLM_EMBEDDING_MODEL``.
            model: Instancia ``SentenceTransformer`` ya cargada (inyectable en
                pruebas para evitar descargar pesos).
        """
        self.settings = settings or get_settings()
        self._model_name = model_name
        self._model: SentenceTransformer | None = model

    @property
    def model_name(self) -> str:
        """Identificador del modelo de embeddings en uso."""
        return self._model_name or self.settings.LLM_EMBEDDING_MODEL

    @property
    def model(self) -> SentenceTransformer:
        """Carga (y cachea) el modelo ``SentenceTransformer`` configurado.

        La importación y la descarga de pesos se difieren hasta la primera
        llamada; si el modelo no está disponible, se propaga la excepción
        subyacente y el llamador debe degradar de forma controlada.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        """Genera el vector de embedding normalizado de un texto.

        Raises:
            EmbeddingError: si el modelo no produce un vector.
        """
        try:
            vector = self.model.encode(text, normalize_embeddings=True)
        except Exception as exc:  # noqa: BLE001 - fallo del modelo local
            raise EmbeddingError(f"fallo al generar el embedding: {exc}") from exc
        if vector is None or len(vector) == 0:
            raise EmbeddingError("el modelo no devolvió ningún vector de embedding")
        return [float(x) for x in vector]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """Genera los vectores de embedding de una lista de textos, por lotes.

        Devuelve una lista del mismo tamaño que ``texts``.
        """
        if not texts:
            return []
        try:
            vectors = self.model.encode(list(texts), normalize_embeddings=True)
        except Exception as exc:  # noqa: BLE001 - fallo del modelo local
            raise EmbeddingError(f"fallo al generar los embeddings: {exc}") from exc
        return [[float(x) for x in vector] for vector in vectors]
