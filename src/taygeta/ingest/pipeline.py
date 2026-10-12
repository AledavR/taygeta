"""Orquestación del pipeline híbrido de ingesta (secciones 4.1-4.4).

Flujo:

1. **Preparación** (:mod:`image_extractor`): rasterización a 300 DPI, detección
   de figuras y recorte/almacenamiento en SeaweedFS.
2. **Pase 1 — OCR local** (:mod:`ocr_client`): transcripción fiel a Markdown/TeX.
3. **Pase 2 — Estructuración remota** (:class:`Structurer`): convierte el texto
   en objetos :class:`~taygeta.core.schema.QuestionRaw` usando ``instructor`` con
   Pydantic v2, sin resolver el ejercicio.
4. **Validación en dos capas** (:mod:`validator`): heurística local + semántica
   remota. El resultado se clasifica de forma estricta como ``VALID`` o
   ``NEEDS_ATTENTION`` y queda en staging (nunca se guarda en la base definitiva
   sin confirmación humana).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from taygeta.core.schema import (
    PLACEHOLDER_PATTERN,
    Choice,
    FigureReference,
    QuestionRaw,
    TranscriptionIntermediate,
    deterministic_question_id,
)
from taygeta.core.settings import Settings, get_settings
from taygeta.ingest.image_extractor import (
    DEFAULT_DPI,
    assign_figure_ids,
    detect_figures,
    extract_and_save_figures,
    open_document,
    rasterize_document,
    upload_crops,
    working_directory,
)
from taygeta.ingest.ocr_client import OCRClient
from taygeta.ingest.validator import (
    STATUS_NEEDS_ATTENTION,
    STATUS_VALID,
    LocalValidationResult,
    LogicalValidationResult,
    RemoteValidationError,
    SemanticValidator,
    validate_local,
)

logger = logging.getLogger(__name__)

__all__ = [
    "STATUS_VALID",
    "STATUS_NEEDS_ATTENTION",
    "IngestPipeline",
    "QuestionDraft",
    "StagedQuestion",
    "StagingResult",
    "StructuredQuestion",
    "Structurer",
    "StructuringError",
]

_PLACEHOLDER_RE = re.compile(PLACEHOLDER_PATTERN)

_STRUCTURING_SYSTEM_PROMPT = """Eres un estructurador de preguntas de exámenes \
científicos. Convierte el texto transcrito por OCR en una lista de preguntas \
estructuradas. Reglas:

1. Identifica los límites entre preguntas y alternativas SIN modificar su sentido.
2. Mantén las expresiones matemáticas como texto TeX/Markdown ($...$ y $$...$$).
3. Conserva los placeholders de figura {{fig:N}} exactamente en su posición.
4. Asocia cada placeholder únicamente a las figuras declaradas en los metadatos.
5. Completa question_type según el formato visible: multiple_choice (si hay \
alternativas A/B/C/D...), open_ended, true_false o numerical.
6. Registra las incertidumbres de lectura en notation_notes; no las resuelvas \
inventando texto.
7. NO resuelvas el ejercicio, NO añadas soluciones, respuestas correctas ni \
explicaciones.
"""


class QuestionDraft(BaseModel):
    """Estructura mínima que el LLM devuelve por pregunta (sin id ni figuras).

    El ``id`` y la asociación de figuras se asignan localmente, de forma
    determinista, para no depender de que el modelo genere UUIDs correctos.
    """

    model_config = ConfigDict(extra="forbid")

    statement_raw: str = Field(..., min_length=1)
    question_type: Literal["multiple_choice", "open_ended", "true_false", "numerical"]
    choices: list[Choice] | None = Field(default=None)
    notation_notes: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class StructuredQuestion:
    """Resultado por pregunta del Pase 2."""

    question: QuestionRaw | None
    error: str | None = None


class StructuringError(RuntimeError):
    """Error controlado durante la estructuración remota (Pase 2)."""


class Structurer:
    """Pase 2: estructura la salida del OCR en ``QuestionRaw`` usando instructor."""

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
        """Convierte la transcripción intermedia en preguntas estructuradas.

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
            "\nDevuelve la lista de preguntas estructuradas, una por cada pregunta del documento."
        )
        return "\n".join(lines)

    def _assemble(
        self,
        draft: QuestionDraft,
        index: int,
        intermediate: TranscriptionIntermediate,
    ) -> StructuredQuestion:
        question_id = deterministic_question_id(
            intermediate.source_reference, *[str(page) for page in intermediate.pages], str(index)
        )
        referenced = self._referenced_figure_ids(draft)
        figures = [figure for figure in intermediate.figures if figure.id in referenced]
        try:
            question = QuestionRaw(
                id=question_id,
                statement_raw=draft.statement_raw,
                question_type=draft.question_type,
                choices=draft.choices,
                figures=figures,
                notation_notes=draft.notation_notes,
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


# ---------------------------------------------------------------------------
# Resultados de staging
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StagedQuestion:
    """Pregunta clasificada tras la validación en dos capas."""

    question: QuestionRaw | None
    status: Literal["VALID", "NEEDS_ATTENTION"]
    reasons: list[str] = field(default_factory=list)
    local: LocalValidationResult | None = None
    logical: LogicalValidationResult | None = None


@dataclass(slots=True)
class StagingResult:
    """Resultado completo de una ingesta, listo para revisión humana."""

    source_reference: str
    status: Literal["VALID", "NEEDS_ATTENTION"]
    questions: list[StagedQuestion] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    session_id: str | None = None
    intermediate: TranscriptionIntermediate | None = None

    @property
    def valid_count(self) -> int:
        return sum(1 for item in self.questions if item.status == STATUS_VALID)

    @property
    def attention_count(self) -> int:
        return sum(1 for item in self.questions if item.status == STATUS_NEEDS_ATTENTION)

    def summary(self) -> dict[str, Any]:
        """Resumen legible del resultado de la ingesta."""
        return {
            "source_reference": self.source_reference,
            "status": self.status,
            "session_id": self.session_id,
            "total": len(self.questions),
            "valid": self.valid_count,
            "needs_attention": self.attention_count,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class IngestPipeline:
    """Orquesta el doble pase de ingesta y la validación en dos capas."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        ocr: Any | None = None,
        structurer: Structurer | None = None,
        semantic_validator: SemanticValidator | None = None,
        storage: Any | None = None,
        database: Any | None = None,
    ) -> None:
        """Crea el pipeline.

        ``ocr``, ``structurer``, ``semantic_validator``, ``storage`` y
        ``database`` son inyectables para pruebas unitarias; por defecto se
        construyen a partir de la configuración global.
        """
        self.settings = settings or get_settings()
        self.ocr = ocr if ocr is not None else OCRClient(self.settings)
        self.structurer = structurer if structurer is not None else Structurer(self.settings)
        self.semantic = (
            semantic_validator
            if semantic_validator is not None
            else SemanticValidator(self.settings)
        )
        self.storage = storage
        self.database = database

    def ingest(
        self,
        source_path: str | Path,
        *,
        hints: str | None = None,
        persist_staging: bool = True,
        session_id: str | None = None,
        upload_figures: bool = True,
    ) -> StagingResult:
        """Ejecuta la ingesta completa y devuelve un resultado de staging.

        Los fallos de OCR, estructuración o validación se convierten en un
        resultado ``NEEDS_ATTENTION`` explicable, nunca se pierden en silencio.
        """
        source = Path(source_path)
        source_reference = source.name
        warnings: list[str] = []

        try:
            rasters = rasterize_document(source, dpi=DEFAULT_DPI)
        except Exception as exc:  # noqa: BLE001
            return self._finish(
                StagingResult(
                    source_reference=source_reference,
                    status=STATUS_NEEDS_ATTENTION,
                    questions=[
                        StagedQuestion(
                            question=None,
                            status=STATUS_NEEDS_ATTENTION,
                            reasons=[f"no se pudo rasterizar el documento: {exc}"],
                        )
                    ],
                    warnings=[f"rasterización fallida: {exc}"],
                ),
                persist_staging,
                session_id,
            )

        figures = self._detect_and_store_figures(source, upload_figures, warnings)

        raw_parts: list[str] = []
        for raster in rasters:
            figure_hint = self._figure_hint(figures, raster.page_index)
            combined_hint = "\n".join(part for part in (figure_hint, hints) if part) or None
            try:
                ocr_result = self.ocr.transcribe_page(
                    raster.png_bytes,
                    page_index=raster.page_index,
                    hints=combined_hint,
                    source_reference=source_reference,
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"OCR de la página {raster.page_index} falló: {exc}")
                raw_parts.append(f"<!-- página {raster.page_index} -->\n[ILEGIBLE]")
                continue
            warnings.extend(ocr_result.warnings)
            raw_parts.append(f"<!-- página {raster.page_index} -->\n{ocr_result.text}")

        intermediate = TranscriptionIntermediate(
            source_reference=source_reference,
            raw_markdown="\n\n".join(raw_parts),
            pages=[raster.page_index for raster in rasters],
            figures=figures,
            warnings=warnings,
        )

        try:
            structured = self.structurer.structure(intermediate, hints=hints)
        except StructuringError as exc:
            return self._finish(
                StagingResult(
                    source_reference=source_reference,
                    status=STATUS_NEEDS_ATTENTION,
                    questions=[
                        StagedQuestion(
                            question=None, status=STATUS_NEEDS_ATTENTION, reasons=[str(exc)]
                        )
                    ],
                    warnings=[*warnings, str(exc)],
                ),
                persist_staging,
                session_id,
            )

        staged: list[StagedQuestion] = []
        for item in structured:
            if item.question is None:
                staged.append(
                    StagedQuestion(
                        question=None,
                        status=STATUS_NEEDS_ATTENTION,
                        reasons=[item.error or "estructuración sin pregunta"],
                    )
                )
                continue
            staged.append(self.validate_question(item.question, max_pages=len(rasters)))

        overall = (
            STATUS_VALID
            if staged and all(item.status == STATUS_VALID for item in staged)
            else STATUS_NEEDS_ATTENTION
        )
        result = StagingResult(
            source_reference=source_reference,
            status=overall,
            questions=staged,
            warnings=warnings,
            intermediate=intermediate,
        )
        return self._finish(result, persist_staging, session_id)

    def validate_question(
        self, question: QuestionRaw, *, max_pages: int | None = None
    ) -> StagedQuestion:
        """Ejecuta las dos capas de validación sobre una pregunta y la clasifica."""
        local = validate_local(question, max_pages=max_pages)
        reasons: list[str] = []

        logical: LogicalValidationResult | None = None
        try:
            logical = self.semantic.validate(question)
        except RemoteValidationError as exc:
            reasons.append(f"validación semántica no disponible: {exc}")
        else:
            if not logical.is_valid:
                reasons.append(logical.rejection_reason or "motivo no especificado")

        if not local.is_valid:
            reasons.append(
                "validación local: " + "; ".join(issue.message for issue in local.errors)
            )

        valid = local.is_valid and logical is not None and logical.is_valid
        return StagedQuestion(
            question=question,
            status=STATUS_VALID if valid else STATUS_NEEDS_ATTENTION,
            reasons=reasons,
            local=local,
            logical=logical,
        )

    def refine_question(
        self,
        intermediate: TranscriptionIntermediate,
        question_index: int,
        *,
        hints: str | None = None,
    ) -> StagedQuestion:
        """Reanaliza una pregunta con contexto correctivo y la revalida.

        Vuelve a ejecutar el Pase 2 (instructor) sobre el texto transcrito con el
        ``hints`` del revisor y aplica las dos capas de validación. No resuelve
        el ejercicio.
        """
        structured = self.structurer.structure(intermediate, hints=hints)
        if question_index < 0 or question_index >= len(structured):
            raise StructuringError(
                f"índice de pregunta {question_index} fuera de rango (0..{len(structured) - 1})"
            )
        item = structured[question_index]
        if item.question is None:
            return StagedQuestion(
                question=None,
                status=STATUS_NEEDS_ATTENTION,
                reasons=[item.error or "el reanálisis no produjo una pregunta válida"],
            )
        max_pages = len(intermediate.pages) or None
        return self.validate_question(item.question, max_pages=max_pages)

    # -- Internos -----------------------------------------------------------
    def _detect_and_store_figures(
        self, source: Path, upload: bool, warnings: list[str]
    ) -> list[FigureReference]:
        detected: list[FigureReference] = []
        try:
            with open_document(source) as document:
                for page in document:
                    detected.extend(detect_figures(page))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"no se pudieron detectar figuras: {exc}")
            return []

        if not detected:
            return []

        detected = assign_figure_ids(detected)
        if self.storage is None or not upload:
            return detected

        figures: list[FigureReference] = []
        try:
            with working_directory() as tmpdir:
                by_page: dict[int, list[FigureReference]] = {}
                for figure in detected:
                    by_page.setdefault(figure.page, []).append(figure)
                for page_index, page_figures in sorted(by_page.items()):
                    crops = extract_and_save_figures(
                        source, page_index, page_figures, tmpdir, dpi=DEFAULT_DPI
                    )
                    crops = upload_crops(crops, self.storage, prefix="figures")
                    for crop in crops:
                        figures.append(
                            FigureReference(
                                id=crop.figure_id,
                                page=crop.page_index,
                                bbox=crop.normalized_bbox,
                                caption=crop.caption,
                                asset_path=crop.asset_uri,
                            )
                        )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"no se pudieron almacenar las figuras definitivas: {exc}")
            return detected
        return figures

    @staticmethod
    def _figure_hint(figures: list[FigureReference], page_index: int) -> str:
        page_figures = [figure for figure in figures if figure.page == page_index]
        if not page_figures:
            return ""
        lines = ["Figuras detectadas en esta página:"]
        for figure in page_figures:
            ymin, xmin, ymax, xmax = figure.bbox
            lines.append(
                f"- {figure.id}: región normalizada [ymin={ymin}, xmin={xmin}, "
                f"ymax={ymax}, xmax={xmax}]"
            )
        lines.append(
            "Inserta el placeholder {{fig:N}} de cada figura en el punto exacto "
            "del flujo de lectura donde aparece."
        )
        return "\n".join(lines)

    def _staging_data(self, result: StagingResult) -> dict[str, Any]:
        return {
            "source_reference": result.source_reference,
            "status": result.status,
            "warnings": result.warnings,
            "intermediate": (
                result.intermediate.model_dump() if result.intermediate is not None else None
            ),
            "questions": [
                {
                    "status": item.status,
                    "reasons": item.reasons,
                    "question": item.question.model_dump() if item.question else None,
                }
                for item in result.questions
            ],
        }

    def _finish(
        self,
        result: StagingResult,
        persist_staging: bool,
        session_id: str | None,
    ) -> StagingResult:
        if persist_staging and self.database is not None:
            result.session_id = self.database.create_staging_session(
                self._staging_data(result), status=result.status, session_id=session_id
            )
        return result
