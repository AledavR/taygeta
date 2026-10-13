"""Orquestación del pipeline híbrido de ingesta (secciones 4.1-4.4).

Flujo:

1. **Preparación** (:mod:`image_extractor`): rasterización a 300 DPI, detección
   de figuras y recorte/almacenamiento en SeaweedFS.
2. **Pase 1 — OCR local** (:mod:`ocr_client`): transcripción fiel a Markdown/TeX.
3. **Estructuración en un solo pase** (:mod:`taygeta.agents.structurer`): una
   única llamada al LLM convierte el texto en objetos
   :class:`~taygeta.core.schema.EnrichedQuestion` (TeX formateado + ``course``,
   ``topic``, ``subtopic``, ``difficulty`` y ``prerequisites``) usando
   ``instructor`` con Pydantic v2, sin resolver el ejercicio.
4. **Validación en dos capas** (:mod:`validator`): heurística local + semántica
   remota. El resultado se clasifica de forma estricta como ``VALID`` o
   ``NEEDS_ATTENTION`` y queda en staging (nunca se guarda en la base definitiva
   sin confirmación humana).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from taygeta.agents.structurer import Structurer, StructuringError
from taygeta.core.schema import EnrichedQuestion, FigureReference, TranscriptionIntermediate
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
    "StagedQuestion",
    "StagingResult",
    "main",
]


# ---------------------------------------------------------------------------
# Resultados de staging
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StagedQuestion:
    """Pregunta enriquecida y clasificada tras la validación en dos capas."""

    question: EnrichedQuestion | None
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
        self.database = database
        self.ocr = ocr if ocr is not None else OCRClient(self.settings)
        self.structurer = (
            structurer if structurer is not None else Structurer(self.settings, database=database)
        )
        self.semantic = (
            semantic_validator
            if semantic_validator is not None
            else SemanticValidator(self.settings)
        )
        self.storage = storage

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
        self, question: EnrichedQuestion, *, max_pages: int | None = None
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
        question: EnrichedQuestion,
        *,
        hints: str | None = None,
    ) -> StagedQuestion:
        """Reanaliza una pregunta con contexto correctivo y la revalida.

        Reejecuta el pase de estructuración sobre UNA sola pregunta usando el mismo
        esquema ``QuestionDraft``: el LLM devuelve, en la misma respuesta JSON, el
        TeX corregido y la taxonomía re-inferida completa (``course``, ``topic``,
        ``subtopic``, ``difficulty`` y ``prerequisites``). Después aplica las dos
        capas de validación. No resuelve el ejercicio.
        """
        refined = self.structurer.refine(intermediate, question, hints=hints)
        if refined.question is None:
            return StagedQuestion(
                question=None,
                status=STATUS_NEEDS_ATTENTION,
                reasons=[refined.error or "el reanálisis no produjo una pregunta válida"],
            )
        max_pages = len(intermediate.pages) or None
        return self.validate_question(refined.question, max_pages=max_pages)

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
        lines = ["Figuras detectadas por geometría en esta página:"]
        for figure in page_figures:
            ymin, xmin, ymax, xmax = figure.bbox
            lines.append(
                f"- {figure.id}: región normalizada [ymin={ymin}, xmin={xmin}, "
                f"ymax={ymax}, xmax={xmax}]"
            )
        lines.append(
            "No todas estas regiones son necesariamente diagramas útiles: puede "
            "haber fondos, logotipos o páginas escaneadas. Inserta el placeholder "
            "{{fig:N}} SOLO si el enunciado depende explícitamente de un gráfico, "
            "diagrama o ilustración dentro de esa región; si no hay tal figura, no "
            "insertes ningún placeholder ni declares fig:N."
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


def main(argv: list[str] | None = None) -> None:
    """CLI mínima de ingesta (``taygeta-ingest``).

    Procesa un documento con el pipeline híbrido (doble pase + validación en dos
    capas), persiste la sesión de staging e imprime un resumen JSON.
    """
    parser = argparse.ArgumentParser(
        prog="taygeta-ingest",
        description="Ingesta híbrida de un documento PDF/PNG/JPG y resumen de staging.",
    )
    parser.add_argument("source", help="Ruta del documento a procesar")
    parser.add_argument("--hint", default=None, help="Contexto correctivo para el análisis OCR/LLM")
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="No crear la sesión de staging en la base de datos",
    )
    args = parser.parse_args(argv)

    pipeline = IngestPipeline(get_settings())
    result = pipeline.ingest(args.source, hints=args.hint, persist_staging=not args.no_persist)
    print(json.dumps(result.summary(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
