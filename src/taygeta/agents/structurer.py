"""Estructuración de un solo pase (sección 4.3).

Consolida la transcripción del OCR, el formateo TeX y la clasificación
curricular en **UNA sola llamada** al LLM remoto (DeepSeek u otro proveedor
compatible con la API de OpenAI vía ``instructor``). El modelo devuelve, por
pregunta, en una única respuesta JSON:

* ``statement_raw`` en Markdown/TeX (``$...$``, ``$$...$$``, placeholders ``{{fig:N}}``);
* ``question_type`` y ``choices``;
* ``course``, ``topic`` y ``subtopic`` (slugs snake_case);
* ``difficulty`` (entero 1-5);
* ``prerequisites`` (lista de slugs).

Los slugs se declaran con el tipo :data:`~taygeta.core.schema.Slug`, que impone
la regex ``^[a-z0-9_]+$``, y se normalizan (``mode="before"``) para tolerar
acentos, mayúsculas y separadores del LLM sin romper la validación. El ``id`` y
la asociación de figuras se asignan localmente de forma determinista. No
resuelve el ejercicio ni añade soluciones o respuestas correctas.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from taygeta.core.schema import (
    PLACEHOLDER_PATTERN,
    Choice,
    EnrichedQuestion,
    Slug,
    TranscriptionIntermediate,
    compute_canonical_hash,
    deterministic_question_id,
    normalize_slug,
    sanitize_math_delimiters,
    slugify,
    utc_now_iso,
)
from taygeta.core.settings import Settings, get_settings

logger = logging.getLogger(__name__)

__all__ = ["QuestionDraft", "StructuredQuestion", "Structurer", "StructuringError"]

_PLACEHOLDER_RE = re.compile(PLACEHOLDER_PATTERN)

_STRUCTURING_SYSTEM_PROMPT = """Eres un estructurador de preguntas de exámenes \
científicos. A partir del texto transcrito por OCR, devuelve UNA lista de \
preguntas y, en la MISMA respuesta, infiere para cada una tanto el enunciado \
formateado como su clasificación curricular. Reglas:

1. statement_raw: enunciado en Markdown/TeX, conservando los placeholders \
{{fig:N}} exactamente en su posición y asociándolos solo a las figuras \
declaradas. Delimitadores matemáticos OBLIGATORIOS: usa EXCLUSIVAMENTE $...$ \
para matemáticas inline y $$...$$ para matemáticas en bloque; PROHIBIDO \
\\[...\\], \\], \\(...\\), \\( o \\), así como sus variantes escapadas \
(reescríbelos siempre como $$...$$ o $...$ respectivamente).
2. question_type: multiple_choice (si hay alternativas A/B/C/D...), open_ended, \
true_false o numerical.
3. choices: alternativas (mínimo 2 para multiple_choice).
4. notation_notes: incertidumbres de lectura; no las resuelvas inventando texto.
5. course: slug del curso en snake_case (minúsculas, dígitos y guion bajo; sin \
espacios ni tildes).
6. topic: slug del tema dentro del curso (snake_case).
7. subtopic: slug del subtema (snake_case) o null si no aplica.
8. difficulty: entero 1-5 según la COMPLEJIDAD MATEMÁTICA; NO lo fijes en 3.
9. prerequisites: lista de conceptos previos como slugs snake_case.

No resuelvas el ejercicio, no añadas soluciones ni respuestas correctas y no \
inventes datos que no se deduzcan del enunciado.
"""

_REFINING_SYSTEM_PROMPT = """Eres un revisor de preguntas de exámenes científicos. \
Recibes una pregunta ya estructurada y un contexto correctivo del revisor humano. \
Corrige el enunciado aplicando el contexto correctivo y, en la MISMA respuesta, \
re-infiere la clasificación curricular completa. Reglas:

1. statement_raw: corrige el enunciado en Markdown/TeX aplicando el contexto \
correctivo, conservando los placeholders {{fig:N}} exactamente en su posición y \
asociándolos solo a las figuras declaradas. Delimitadores matemáticos \
OBLIGATORIOS: usa EXCLUSIVAMENTE $...$ para matemáticas inline y $$...$$ para \
matemáticas en bloque; PROHIBIDO \\[...\\], \\], \\(...\\), \\( o \\), así como \
sus variantes escapadas (reescríbelos siempre como $$...$$ o $...$ \
respectivamente).
2. question_type: multiple_choice (si hay alternativas A/B/C/D...), open_ended, \
true_false o numerical.
3. choices: alternativas corregidas (mínimo 2 para multiple_choice).
4. notation_notes: incertidumbres de lectura; no las resuelvas inventando texto.
5. course: slug del curso en snake_case (minúsculas, dígitos y guion bajo; sin \
espacios ni tildes), re-inferido del nuevo contexto.
6. topic: slug del tema dentro del curso (snake_case).
7. subtopic: slug del subtema (snake_case) o null si no aplica.
8. difficulty: entero 1-5 según la COMPLEJIDAD MATEMÁTICA; NO lo fijes en 3.
9. prerequisites: lista de conceptos previos como slugs snake_case.

No resuelvas el ejercicio, no añadas soluciones ni respuestas correctas y no \
inventes datos que no se deduzcan del enunciado.
"""


class QuestionDraft(BaseModel):
    """Respuesta única del LLM por pregunta: estructura + clasificación curricular.

    El ``id`` y la asociación de figuras se asignan localmente, de forma
    determinista, para no depender de que el modelo genere UUIDs correctos.
    """

    model_config = ConfigDict(extra="forbid")

    statement_raw: str = Field(..., min_length=1)
    question_type: Literal["multiple_choice", "open_ended", "true_false", "numerical"]
    choices: list[Choice] | None = Field(default=None)
    notation_notes: list[str] = Field(default_factory=list)
    course: Slug = Field(..., description="Slug del curso (regex ^[a-z0-9_]+$).")
    topic: Slug = Field(..., description="Slug del tema (regex ^[a-z0-9_]+$).")
    subtopic: Slug | None = Field(default=None, description="Slug del subtema o null.")
    difficulty: int = Field(..., ge=1, le=5, description="Dificultad estimada 1-5.")
    prerequisites: list[Slug] = Field(
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


@dataclass(slots=True)
class StructuredQuestion:
    """Resultado por pregunta del pase único de estructuración."""

    question: EnrichedQuestion | None
    error: str | None = None


class StructuringError(RuntimeError):
    """Error controlado durante la estructuración remota."""


class Structurer:
    """Estructura la salida del OCR en ``EnrichedQuestion`` en un único pase."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
        model: str | None = None,
        max_retries: int | None = None,
    ) -> None:
        """Crea el estructurador.

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

    def structure(
        self,
        intermediate: TranscriptionIntermediate,
        *,
        hints: str | None = None,
    ) -> list[StructuredQuestion]:
        """Convierte la transcripción intermedia en preguntas enriquecidas.

        Una sola llamada al LLM devuelve, por pregunta, el TeX formateado y la
        clasificación curricular (curso, tema, subtema, dificultad y
        prerrequisitos).

        Raises:
            StructuringError: si no hay texto, el proveedor falla o no devuelve
                ninguna pregunta.
        """
        text = intermediate.raw_markdown.strip()
        if not text:
            raise StructuringError("no hay texto transcrito que estructurar")

        messages = [
            {"role": "system", "content": _STRUCTURING_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_message(intermediate, hints)},
        ]
        try:
            raw = self.client.chat.completions.create(
                model=self.model,
                response_model=list[QuestionDraft],
                messages=messages,
                temperature=0.0,
                max_retries=self._max_retries,
            )
        except Exception as exc:  # noqa: BLE001 - errores de proveedor y de esquema
            raise StructuringError(f"fallo en la estructuración remota: {exc}") from exc

        if raw is None:
            drafts: list[QuestionDraft] = []
        elif isinstance(raw, list):
            drafts = raw
        else:
            drafts = list(raw)
        if not drafts:
            raise StructuringError("el LLM remoto no produjo ninguna pregunta")

        return [self._assemble(draft, index, intermediate) for index, draft in enumerate(drafts)]

    def refine(
        self,
        intermediate: TranscriptionIntermediate,
        current: EnrichedQuestion,
        *,
        hints: str | None = None,
    ) -> StructuredQuestion:
        """Reanaliza UNA pregunta aplicando contexto correctivo.

        Una sola llamada al LLM devuelve, en la MISMA respuesta JSON, el TeX
        corregido y la clasificación curricular re-inferida (``course``, ``topic``,
        ``subtopic``, ``difficulty`` y ``prerequisites``) usando el mismo esquema
        :class:`QuestionDraft`. Conserva el ``id`` y ``created_at`` de la pregunta
        original para que el objeto actualice la base. No resuelve el ejercicio.

        Raises:
            StructuringError: si el proveedor falla o no devuelve una pregunta.
        """
        messages = [
            {"role": "system", "content": _REFINING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._build_refine_user_message(intermediate, current, hints),
            },
        ]
        try:
            raw = self.client.chat.completions.create(
                model=self.model,
                response_model=QuestionDraft,
                messages=messages,
                temperature=0.0,
                max_retries=self._max_retries,
            )
        except Exception as exc:  # noqa: BLE001 - errores de proveedor y de esquema
            raise StructuringError(f"fallo en el refinamiento remoto: {exc}") from exc

        if raw is None:
            raise StructuringError("el LLM remoto no produjo ninguna pregunta refinada")

        return self._assemble(
            raw,
            0,
            intermediate,
            question_id=current.id,
            created_at=current.created_at,
        )

    def _build_user_message(
        self, intermediate: TranscriptionIntermediate, hints: str | None
    ) -> str:
        lines = ["Texto transcrito del OCR (Markdown/TeX):", "", intermediate.raw_markdown]
        lines += [
            "",
            "Metadatos:",
            f"- Origen: {intermediate.source_reference}",
            f"- Páginas: {', '.join(str(p) for p in intermediate.pages) or '(ninguna)'}",
        ]
        if intermediate.figures:
            lines.append("- Figuras disponibles:")
            for figure in intermediate.figures:
                ymin, xmin, ymax, xmax = figure.bbox
                lines.append(
                    f"  - {figure.id} (página {figure.page}, bbox [ymin={ymin}, "
                    f"xmin={xmin}, ymax={ymax}, xmax={xmax}])"
                )
        if hints:
            lines.append(f"\nContexto adicional del revisor: {hints}")
        lines.append(
            "\nDevuelve la lista de preguntas estructuradas, una por pregunta, "
            "incluyendo statement_raw (TeX), question_type, choices, notation_notes, "
            "course, topic, subtopic, difficulty y prerequisites."
        )
        return "\n".join(lines)

    def _build_refine_user_message(
        self,
        intermediate: TranscriptionIntermediate,
        current: EnrichedQuestion,
        hints: str | None,
    ) -> str:
        lines = ["Pregunta actual a corregir (Markdown/TeX):", "", current.statement_raw]
        lines += [
            "",
            "Clasificación curricular actual:",
            f"- course: {current.course}",
            f"- topic: {current.topic}",
            f"- subtopic: {current.subtopic}",
            f"- difficulty: {current.difficulty}",
            f"- prerequisites: {', '.join(current.prerequisites) or '(ninguno)'}",
        ]
        if current.choices:
            lines += ["", "Alternativas actuales:"]
            lines += [f"- {choice.id}: {choice.content_raw}" for choice in current.choices]
        if intermediate.figures:
            lines += ["", "Figuras disponibles:"]
            for figure in intermediate.figures:
                ymin, xmin, ymax, xmax = figure.bbox
                lines.append(
                    f"  - {figure.id} (página {figure.page}, bbox [ymin={ymin}, "
                    f"xmin={xmin}, ymax={ymax}, xmax={xmax}])"
                )
        if hints:
            lines += ["", f"Contexto correctivo del revisor: {hints}"]
        lines.append(
            "\nCorrige el enunciado aplicando el contexto correctivo y RE-INFIERE la "
            "clasificación curricular completa. Devuelve UNA única pregunta con "
            "statement_raw (TeX), question_type, choices, notation_notes, course, topic, "
            "subtopic, difficulty y prerequisites."
        )
        return "\n".join(lines)

    def _assemble(
        self,
        draft: QuestionDraft,
        index: int,
        intermediate: TranscriptionIntermediate,
        *,
        question_id: str | None = None,
        created_at: str | None = None,
    ) -> StructuredQuestion:
        resolved_id = question_id or deterministic_question_id(
            intermediate.source_reference, *[str(page) for page in intermediate.pages], str(index)
        )
        referenced = self._referenced_figure_ids(draft)
        figures = [figure for figure in intermediate.figures if figure.id in referenced]
        # Normaliza los delimitadores matemáticos antes de calcular el hash canónico
        # para que enunciado y deduplicación queden coherentes (idempotente con el
        # validador de ``QuestionRaw``).
        statement_raw = sanitize_math_delimiters(draft.statement_raw)
        try:
            question = EnrichedQuestion(
                id=resolved_id,
                statement_raw=statement_raw,
                question_type=draft.question_type,
                choices=draft.choices,
                figures=figures,
                notation_notes=draft.notation_notes,
                course=draft.course,
                topic=draft.topic,
                subtopic=draft.subtopic,
                difficulty=draft.difficulty,
                prerequisites=draft.prerequisites,
                source_metadata={"source_reference": intermediate.source_reference},
                created_at=created_at or utc_now_iso(),
                canonical_hash=compute_canonical_hash(statement_raw),
            )
        except ValidationError as exc:
            return StructuredQuestion(
                question=None,
                error=f"la pregunta {index + 1} no superó el esquema: {exc}",
            )
        return StructuredQuestion(question=question)

    @staticmethod
    def _referenced_figure_ids(draft: QuestionDraft) -> set[str]:
        text = draft.statement_raw
        if draft.choices:
            text += "\n" + "\n".join(choice.content_raw for choice in draft.choices)
        return {f"fig:{match}" for match in _PLACEHOLDER_RE.findall(text)}
