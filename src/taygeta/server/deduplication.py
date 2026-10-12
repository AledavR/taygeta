"""Deduplicación semántica de preguntas por embeddings (filtro de ingesta).

Flujo:

1. Al ingerir una pregunta se genera su embedding con el modelo configurado
   (:class:`~taygeta.agents.embeddings.Embedder`).
2. Se compara (similitud coseno) contra los embeddings ya almacenados en la base
   de datos, restringido al mismo curso.
3. Si alguna similitud supera ``DUPLICATE_SIMILARITY_THRESHOLD``, la pregunta
   queda marcada como candidata a duplicado.
4. Opcionalmente, un juez LLM decide si la nueva y la candidata son
   matemáticamente la misma pregunta (ignorando variaciones de notación/OCR).
   Si no hay juez disponible, se usa el candidato de mayor similitud.
5. Si se confirma el duplicado, se descarta el texto nuevo y se fusionan los
   metadatos (orígenes ``source_reference`` y prerrequisitos) en la pregunta
   existente. Nunca se almacenan soluciones ni respuestas correctas.

La generación de embeddings es local (``sentence-transformers``) y el juez LLM
degradan de forma controlada: si el modelo local o el proveedor no están
disponibles, la ingesta continúa y se conserva únicamente la deduplicación por
hash exacto de :meth:`Database.insert_question`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from taygeta.agents.embeddings import Embedder, question_embedding_text
from taygeta.core.schema import EnrichedQuestion
from taygeta.core.settings import Settings, get_settings
from taygeta.server.db import Database

if TYPE_CHECKING:
    import instructor  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = [
    "DedupOutcome",
    "DeduplicationService",
    "DuplicateJudgement",
    "merge_duplicate_metadata",
]

#: Número máximo de candidatos a enviar al juez LLM.
_MAX_CANDIDATES = 5


class DuplicateJudgement(BaseModel):
    """Veredicto del juez LLM sobre un candidato a duplicado."""

    model_config = ConfigDict(extra="forbid")

    is_duplicate: bool
    matching_index: int = Field(
        default=-1,
        ge=-1,
        description="Índice (0-based) de la candidata duplicada, o -1 si no hay duplicado.",
    )
    reason: str = Field(default="", description="Explicación breve en español.")


@dataclass(slots=True)
class DedupOutcome:
    """Resultado del filtro de deduplicación sobre una pregunta nueva."""

    status: Literal["INSERTED", "DUPLICATE"]
    existing_id: str | None = None
    similarity: float | None = None
    reason: str = ""


_DUPLICATE_JUDGE_SYSTEM_PROMPT = """Eres un juez de deduplicación de preguntas de \
exámenes científicos. Compara la pregunta nueva con las candidatas y determina si \
es MATEMÁTICAMENTE la misma pregunta que alguna de ellas. Ignora diferencias menores \
de notación (variables renombradas como X vs E, cambios de letra o símbolo), errores \
de transcripción OCR y variaciones de redacción que no alteren el enunciado \
matemático. NO resuelvas el ejercicio y NO evalúes respuestas correctas.

Devuelve:
- is_duplicate: true SOLO si la pregunta nueva es la misma pregunta matemática que \
alguna candidata.
- matching_index: el índice (0-based) de la candidata duplicada, o -1 si no hay duplicado.
- reason: breve explicación en español.
"""


def _origin_references(question: EnrichedQuestion) -> list[str]:
    """Recupera los orígenes (``source_reference``) conocidos de una pregunta."""
    metadata = question.source_metadata or {}
    references = [str(ref) for ref in metadata.get("source_references", []) if ref]
    primary = metadata.get("source_reference")
    if primary and str(primary) not in references:
        references.insert(0, str(primary))
    return references


def merge_duplicate_metadata(
    existing: EnrichedQuestion, incoming: EnrichedQuestion
) -> EnrichedQuestion:
    """Fusiona los metadatos de una pregunta duplicada entrante en la existente.

    Conserva el enunciado, alternativas y clasificación de la pregunta ya
    almacenada (la "canónica") y añade los orígenes y prerrequisitos de la nueva.
    No modifica soluciones ni respuestas correctas (el esquema no las contiene).
    """
    references = list(dict.fromkeys(_origin_references(existing) + _origin_references(incoming)))
    source_metadata = dict(existing.source_metadata)
    if references:
        source_metadata["source_reference"] = references[0]
        source_metadata["source_references"] = references
    prerequisites = list(dict.fromkeys([*existing.prerequisites, *incoming.prerequisites]))
    return existing.model_copy(
        update={
            "source_metadata": source_metadata,
            "prerequisites": prerequisites,
        }
    )


class DeduplicationService:
    """Coordina embeddings, similitud y juez LLM para la deduplicación semántica."""

    def __init__(
        self,
        database: Database,
        settings: Settings | None = None,
        *,
        embedder: Embedder | None = None,
        judge_model: str | None = None,
        threshold: float | None = None,
    ) -> None:
        """Crea el servicio.

        Args:
            database: Acceso a la base de datos (preguntas y embeddings).
            settings: Configuración a usar; por defecto la global.
            embedder: :class:`Embedder` ya construido (inyectable en pruebas).
            judge_model: Sobrescribe ``LLM_MODEL`` para el juez de duplicados.
            threshold: Sobrescribe ``DUPLICATE_SIMILARITY_THRESHOLD`` (útil en
                scripts de migración y pruebas).
        """
        self.settings = settings or get_settings()
        self.database = database
        self._embedder = embedder
        self._judge_model = judge_model
        self._threshold = threshold
        self._judge_client: Any | None = None

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self.settings)
        return self._embedder

    @property
    def threshold(self) -> float:
        if self._threshold is not None:
            return self._threshold
        return self.settings.DUPLICATE_SIMILARITY_THRESHOLD

    @property
    def _judge_model_name(self) -> str:
        return self._judge_model or self.settings.LLM_MODEL

    @property
    def _judge_client_instance(self) -> Any:
        if self._judge_client is None:
            import instructor

            self._judge_client = instructor.from_openai(
                OpenAI(
                    base_url=self.settings.LLM_BASE_URL,
                    api_key=self.settings.LLM_API_KEY,
                    timeout=float(self.settings.LLM_TIMEOUT_SECONDS),
                )
            )
        return self._judge_client

    def embed_question(self, question: EnrichedQuestion) -> list[float] | None:
        """Genera el embedding de una pregunta de forma best-effort.

        Devuelve ``None`` (sin elevar) si el modelo local falla al cargar o al
        codificar, para no bloquear la ingesta.
        """
        try:
            return self.embedder.embed(question_embedding_text(question))
        except Exception:  # noqa: BLE001 - la deduplicación semántica es opcional
            logger.warning(
                "no se pudo generar el embedding de la pregunta %s", question.id, exc_info=True
            )
            return None

    def ingest(self, question: EnrichedQuestion) -> DedupOutcome:
        """Ejecuta el filtro de ingesta y devuelve el resultado.

        Genera el embedding, busca candidatos por similitud, consulta al juez y,
        si confirma un duplicado, fusiona los metadatos en la pregunta existente.
        En caso contrario delega en :meth:`Database.insert_question`.
        """
        embedding = self.embed_question(question)
        if embedding is not None:
            matches = self.database.find_similar_questions(
                embedding,
                threshold=self.threshold,
                course=question.course,
                limit=_MAX_CANDIDATES,
            )
            if matches:
                candidates, similarities, ids = self._load_candidates(matches)
                if candidates:
                    match_index = self._resolve_candidate(question, candidates)
                    if match_index is not None:
                        existing_id = ids[match_index]
                        self._merge_into(existing_id, question)
                        return DedupOutcome(
                            status="DUPLICATE",
                            existing_id=existing_id,
                            similarity=similarities[match_index],
                            reason="pregunta duplicada (similitud semántica)",
                        )
        status = self.database.insert_question(question, embedding=embedding)
        return DedupOutcome(status=status)

    def judge(
        self, new_question: EnrichedQuestion, candidates: list[EnrichedQuestion]
    ) -> DuplicateJudgement | None:
        """Consulta al juez LLM si la nueva es duplicado de alguna candidata.

        Devuelve ``None`` si no hay candidatas, el LLM no está configurado o el
        juez falla; en ese caso el llamador puede caer al umbral de similitud.
        """
        if not candidates or not self.settings.llm_configured:
            return None
        messages = [
            {"role": "system", "content": _DUPLICATE_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_judge_user_message(new_question, candidates)},
        ]
        try:
            return self._judge_client_instance.chat.completions.create(
                model=self._judge_model_name,
                response_model=DuplicateJudgement,
                messages=messages,
                temperature=0.0,
            )
        except Exception:  # noqa: BLE001 - el juez es opcional; se degrada al umbral
            logger.warning(
                "el juez de duplicados falló; se usa el umbral de similitud", exc_info=True
            )
            return None

    def _load_candidates(
        self, matches: list[tuple[str, float]]
    ) -> tuple[list[EnrichedQuestion], list[float], list[str]]:
        candidates: list[EnrichedQuestion] = []
        similarities: list[float] = []
        ids: list[str] = []
        for question_id, similarity in matches:
            candidate = self.database.get_question(question_id)
            if candidate is None:
                continue
            candidates.append(candidate)
            similarities.append(similarity)
            ids.append(question_id)
        return candidates, similarities, ids

    def _resolve_candidate(
        self, new_question: EnrichedQuestion, candidates: list[EnrichedQuestion]
    ) -> int | None:
        judgement = self.judge(new_question, candidates)
        if judgement is not None:
            if judgement.is_duplicate and 0 <= judgement.matching_index < len(candidates):
                return judgement.matching_index
            return None
        # Sin juez (no configurado o fallo): el candidato de mayor similitud.
        return 0

    def _merge_into(self, existing_id: str, incoming: EnrichedQuestion) -> None:
        existing = self.database.get_question(existing_id)
        if existing is None:
            return
        merged = merge_duplicate_metadata(existing, incoming)
        # El enunciado no cambia, por lo que el embedding existente sigue siendo válido.
        self.database.update_question(merged)

    @staticmethod
    def _build_judge_user_message(
        new_question: EnrichedQuestion, candidates: list[EnrichedQuestion]
    ) -> str:
        lines = ["Pregunta nueva a evaluar (Markdown/TeX):", "", new_question.statement_raw]
        if new_question.choices:
            lines += ["", "Alternativas:"]
            lines += [f"- {choice.id}: {choice.content_raw}" for choice in new_question.choices]
        lines += ["", "Candidatas ya existentes:"]
        for index, candidate in enumerate(candidates):
            lines += [f"[{index}] {candidate.statement_raw}"]
            if candidate.choices:
                alternatives = " | ".join(
                    f"{choice.id}: {choice.content_raw}" for choice in candidate.choices
                )
                lines.append(f"    Alternativas: {alternatives}")
        lines += [
            "",
            "Decide si la pregunta nueva es MATEMÁTICAMENTE la misma que alguna de las "
            "candidatas (ignorando diferencias menores de notación, OCR o redacción). "
            "Devuelve is_duplicate y matching_index.",
        ]
        return "\n".join(lines)
