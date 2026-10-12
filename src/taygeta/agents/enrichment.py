"""Enriquecimiento curricular automático mediante el LLM remoto.

A partir del contenido TeX/Markdown de una pregunta, el modelo infiere
**obligatoriamente**:

* ``course``, ``topic`` y ``subtopic`` (slugs en ``snake_case``);
* ``difficulty``: entero 1-5 estimado por la complejidad matemática (no se fija
  en 3);
* ``prerequisites``: lista de conceptos previos como slugs.

Todas las entradas se normalizan con :func:`taygeta.core.schema.normalize_slug`
para cumplir la regex ``^[a-z0-9_]+$`` y evitar cadenas vacías.

La interfaz es independiente del motor (se usa ``instructor`` sobre un proveedor
compatible con la API de OpenAI configurado con ``LLM_*``). No resuelve el
ejercicio ni añade soluciones o respuestas correctas.
"""

from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator

from taygeta.core.schema import QuestionRaw, normalize_slug, slugify
from taygeta.core.settings import Settings, get_settings

logger = logging.getLogger(__name__)

__all__ = ["EnrichmentError", "EnrichmentResult", "QuestionEnricher"]

_ENRICHMENT_SYSTEM_PROMPT = """Eres un clasificador curricular de ejercicios \
científicos. A partir del enunciado (Markdown/TeX) infiere OBLIGATORIAMENTE:

1. `course`: slug del curso en snake_case (minúsculas, dígitos y guion bajo; sin \
espacios ni tildes).
2. `topic`: slug del tema dentro del curso (snake_case).
3. `subtopic`: slug del subtema (snake_case) o null si no aplica.
4. `difficulty`: entero de 1 a 5 estimado según la COMPLEJIDAD MATEMÁTICA \
(1 = trivial, 5 = muy complejo). Estímalo de forma realista; NO lo fijes en 3.
5. `prerequisites`: lista de conceptos previos necesarios, cada uno como slug \
snake_case (p. ej. ["espacios_vectoriales", "normas"]).

No resuelvas el ejercicio, no añadas soluciones ni respuestas correctas y no \
inventes datos que no se deduzcan del enunciado.
"""


class EnrichmentError(RuntimeError):
    """Error controlado del enriquecimiento remoto."""


class EnrichmentResult(BaseModel):
    """Clasificación curricular inferida para una pregunta."""

    model_config = ConfigDict(extra="forbid")

    course: str = Field(..., description="Slug del curso.")
    topic: str = Field(..., description="Slug del tema.")
    subtopic: str | None = Field(default=None, description="Slug del subtema o null.")
    difficulty: int = Field(..., ge=1, le=5, description="Dificultad estimada 1-5.")
    prerequisites: list[str] = Field(
        default_factory=list, description="Slugs de conceptos previos."
    )

    @field_validator("course", "topic", mode="before")
    @classmethod
    def _normalize_required(cls, value: Any) -> str:
        return normalize_slug(value if isinstance(value, str) else None)

    @field_validator("subtopic", mode="before")
    @classmethod
    def _normalize_subtopic(cls, value: Any) -> str | None:
        return slugify(value if isinstance(value, str) else None) or None

    @field_validator("prerequisites", mode="before")
    @classmethod
    def _normalize_prerequisites(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = value.replace(";", ",").split(",")
        if not isinstance(value, (list, tuple, set)):
            return []
        result: list[str] = []
        for item in value:
            slug = slugify(str(item))
            if slug and slug not in result:
                result.append(slug)
        return result

    @field_validator("difficulty", mode="before")
    @classmethod
    def _coerce_difficulty(cls, value: Any) -> int:
        try:
            difficulty = int(float(value))
        except (TypeError, ValueError):
            return 3
        return max(1, min(5, difficulty))


class QuestionEnricher:
    """Infiere la clasificación curricular de una pregunta con el LLM remoto."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
        model: str | None = None,
        max_retries: int | None = None,
    ) -> None:
        """Crea el enriquecedor.

        Args:
            settings: Configuración a usar; por defecto la global.
            client: Cliente ``instructor`` ya construido (inyectable en pruebas).
            model: Sobrescribe ``LLM_MODEL``.
        """
        self.settings = settings or get_settings()
        self._client = client
        self._model = model
        self._max_retries = (
            max_retries if max_retries is not None else self.settings.LLM_MAX_RETRIES
        )

    @property
    def model(self) -> str:
        return self._model or self.settings.LLM_MODEL

    @property
    def client(self) -> Any:
        if self._client is None:
            import instructor

            self._client = instructor.from_openai(self._openai_client())
        return self._client

    def _openai_client(self) -> OpenAI:
        return OpenAI(
            base_url=self.settings.LLM_BASE_URL,
            api_key=self.settings.LLM_API_KEY,
            timeout=float(self.settings.LLM_TIMEOUT_SECONDS),
            max_retries=self._max_retries,
        )

    def enrich(self, question: QuestionRaw, *, hints: str | None = None) -> EnrichmentResult:
        """Infiere la clasificación curricular de ``question``.

        Raises:
            EnrichmentError: si el proveedor falla o la respuesta no es válida.
        """
        text = question.statement_raw
        if question.choices:
            alternatives = "\n".join(
                f"{choice.id}: {choice.content_raw}" for choice in question.choices
            )
            text += f"\n\nAlternativas:\n{alternatives}"
        user_text = "Enunciado (Markdown/TeX):\n\n" + text
        if hints:
            user_text += f"\n\nContexto del revisor: {hints}"
        user_text += (
            "\n\nClasifica el ejercicio y devuelve course, topic, subtopic, "
            "difficulty y prerequisites."
        )

        try:
            result = self.client.chat.completions.create(
                model=self.model,
                response_model=EnrichmentResult,
                messages=[
                    {"role": "system", "content": _ENRICHMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.0,
                max_retries=self._max_retries,
            )
        except Exception as exc:  # noqa: BLE001 - errores de proveedor y de esquema
            raise EnrichmentError(f"fallo en el enriquecimiento remoto: {exc}") from exc
        return result
