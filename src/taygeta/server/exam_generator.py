"""Generación de exámenes de práctica (RAG, Typst, ZIP) — sección 9.3.

Responsabilidades:

* ``retrieve_questions``: recupera ejercicios reales por curso/tema/dificultad
  usando consultas parametrizadas de SQLite (FTS5 + filtros).
* ``build_exam_typst`` / ``build_exam_latex``: ensamblan la plantilla a partir de
  los enunciados (Markdown/TeX) mediante el transpilador.
* ``copy_assets``: descarga desde SeaweedFS **solo** las figuras utilizadas y las
  copia al directorio ``assets/`` del workspace de compilación.
* ``generate_exam``: recupera, ensambla, copia assets y compila el PDF con el CLI
  nativo de Typst; devuelve ``exam_id`` único, resumen y URL temporal.
* ``build_source_zip``: empaqueta ``main.typ`` (o ``main.tex``), ``assets/`` y
  ``README.txt`` en un ZIP para su descarga.

No se generan ni incluyen soluciones o respuestas correctas.
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from taygeta.core import transpiler
from taygeta.core.schema import EnrichedQuestion, FigureReference
from taygeta.core.settings import Settings, get_settings
from taygeta.core.urls import split_asset_reference
from taygeta.server.db import Database
from taygeta.server.object_storage import ObjectStorage

logger = logging.getLogger(__name__)

__all__ = ["ExamGenerator", "ExamGenerationError", "ExamResult"]

_ALLOWED_FORMATS = frozenset({"typst", "latex"})
_HEX_CHARS = frozenset("0123456789abcdef")


class ExamGenerationError(RuntimeError):
    """Error controlado de la generación o empaquetado de exámenes."""


def _is_valid_exam_id(exam_id: str) -> bool:
    return len(exam_id) == 32 and all(char in _HEX_CHARS for char in exam_id)


@dataclass(slots=True)
class ExamResult:
    """Resultado de una generación de examen."""

    exam_id: str
    status: str
    num_questions: int
    pdf_path: Path | None
    pdf_url: str | None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "exam_id": self.exam_id,
            "status": self.status,
            "num_questions": self.num_questions,
            "pdf_url": self.pdf_url,
            "warnings": self.warnings,
        }


class ExamGenerator:
    """Motor de generación y empaquetado de exámenes de práctica."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        database: Database | None = None,
        storage: ObjectStorage | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.database = database or Database(settings=self.settings)
        self.storage = storage or ObjectStorage(self.settings)

    # -- Recuperación (RAG) -------------------------------------------------
    def retrieve_questions(
        self,
        *,
        course: str | None = None,
        topics: Sequence[str] | None = None,
        difficulty: int | None = None,
        limit: int = 10,
    ) -> list[EnrichedQuestion]:
        """Recupera preguntas filtradas por curso, temas y dificultad.

        Usa consultas parametrizadas de ``Database`` y un límite acotado. No
        devuelve soluciones ni respuestas correctas.
        """
        limit = max(1, min(limit, 100))
        if topics:
            seen: dict[str, EnrichedQuestion] = {}
            for topic in topics:
                for question in self.database.list_questions(
                    course=course, topic=topic, difficulty=difficulty, limit=limit
                ):
                    seen[question.id] = question
            return list(seen.values())[:limit]
        return self.database.list_questions(course=course, difficulty=difficulty, limit=limit)

    # -- Nombrado local de figuras ------------------------------------------
    @staticmethod
    def local_asset_name(figure: FigureReference) -> str:
        """Nombre de archivo local y seguro para una figura (relativo a ``assets/``)."""
        suffix = Path(figure.asset_path or "").suffix or ".png"
        if not suffix.startswith("."):
            suffix = "." + suffix
        return figure.id.replace(":", "_") + suffix

    def figure_paths_map(self, questions: Sequence[EnrichedQuestion]) -> dict[str, str]:
        """Mapea ``fig:id`` → nombre de asset local para el transpilador."""
        mapping: dict[str, str] = {}
        for question in questions:
            for figure in question.figures:
                mapping[figure.id] = self.local_asset_name(figure)
        return mapping

    # -- Plantillas ---------------------------------------------------------
    def build_exam_typst(
        self,
        questions: Sequence[EnrichedQuestion],
        *,
        title: str = "Examen de práctica",
        figure_paths: Mapping[str, str] | None = None,
    ) -> str:
        """Ensambla la fuente Typst del examen a partir de los enunciados."""
        paths = dict(figure_paths or self.figure_paths_map(questions))
        lines = [f"= {title}", ""]
        for index, question in enumerate(questions, start=1):
            lines.append(f"== Pregunta {index}")
            lines.append(
                transpiler.markdown_to_typst(question.statement_raw, figure_paths=paths).rstrip()
            )
            lines.append("")
        return "\n".join(lines) + "\n"

    def build_exam_latex(
        self,
        questions: Sequence[EnrichedQuestion],
        *,
        title: str = "Examen de práctica",
        figure_paths: Mapping[str, str] | None = None,
    ) -> str:
        """Ensambla la fuente LaTeX del examen usando ``\\includegraphics``."""
        paths = dict(figure_paths or self.figure_paths_map(questions))
        lines = [
            "\\documentclass{article}",
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage{graphicx}",
            "\\usepackage{amsmath,amssymb}",
            "\\usepackage[spanish]{babel}",
            "\\begin{document}",
            f"\\title{{{title}}}",
            "\\maketitle",
        ]
        for index, question in enumerate(questions, start=1):
            lines.append(f"\\section*{{Pregunta {index}}}")
            lines.append(
                transpiler.markdown_to_latex(question.statement_raw, figure_paths=paths).rstrip()
            )
        lines.append("\\end{document}")
        return "\n".join(lines) + "\n"

    # -- Assets -------------------------------------------------------------
    def copy_assets(
        self, questions: Sequence[EnrichedQuestion], assets_dir: str | Path
    ) -> dict[str, str]:
        """Descarga desde SeaweedFS las figuras utilizadas a ``assets_dir``.

        Solo copia las figuras referenciadas por los enunciados; no guarda una
        segunda copia local permanente.
        """
        directory = Path(assets_dir)
        directory.mkdir(parents=True, exist_ok=True)
        mapping: dict[str, str] = {}
        for question in questions:
            for figure in question.figures:
                if not figure.asset_path:
                    continue
                local_name = self.local_asset_name(figure)
                mapping[figure.id] = local_name
                destination = directory / local_name
                if destination.exists():
                    continue
                bucket, key = split_asset_reference(
                    figure.asset_path, default_bucket=self.settings.S3_BUCKET
                )
                self.storage.download_to_file(key, destination, bucket=bucket)
        return mapping

    # -- Generación ---------------------------------------------------------
    def generate_exam(
        self,
        *,
        course: str | None = None,
        topics: Sequence[str] | None = None,
        difficulty: int | None = None,
        num_questions: int = 10,
        exam_id: str | None = None,
        title: str | None = None,
        downloads_dir: str | Path | None = None,
    ) -> ExamResult:
        """Genera y compila un examen de práctica completo.

        Informa explícitamente si no se recuperaron suficientes preguntas.
        """
        downloads = Path(downloads_dir or self.settings.DOWNLOADS_DIR)
        exams_dir = downloads / "exams"
        exams_dir.mkdir(parents=True, exist_ok=True)

        questions = self.retrieve_questions(
            course=course, topics=topics, difficulty=difficulty, limit=num_questions
        )
        warnings: list[str] = []
        status = "ok"
        if len(questions) < num_questions:
            status = "partial"
            warnings.append(
                f"solo se encontraron {len(questions)} de las {num_questions} preguntas solicitadas"
            )
        if not questions:
            return ExamResult(exam_id or "", "error", 0, None, None, warnings)

        identifier = exam_id or uuid.uuid4().hex
        if not _is_valid_exam_id(identifier):
            raise ExamGenerationError("exam_id inválido")

        source_dir = exams_dir / identifier
        assets_dir = source_dir / "assets"
        source_dir.mkdir(parents=True, exist_ok=True)
        assets_dir.mkdir(parents=True, exist_ok=True)

        figure_paths = self.copy_assets(questions, assets_dir)
        if title:
            resolved_title = title
        elif course:
            resolved_title = f"Examen de práctica — {course}"
        else:
            resolved_title = "Examen de práctica"
        typst = self.build_exam_typst(questions, title=resolved_title, figure_paths=figure_paths)
        (source_dir / "main.typ").write_text(typst)
        manifest = {
            "exam_id": identifier,
            "title": resolved_title,
            "figure_paths": figure_paths,
            "questions": [question.model_dump(mode="json") for question in questions],
        }
        (source_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))

        compile_result = transpiler.compile_typst(typst, source_dir, output_name="output.pdf")
        if not compile_result.ok:
            return ExamResult(
                identifier,
                "error",
                len(questions),
                None,
                None,
                [compile_result.format_errors()],
            )

        pdf_path = exams_dir / f"{identifier}.pdf"
        if compile_result.pdf_path is not None:
            shutil.move(str(compile_result.pdf_path), str(pdf_path))
        pdf_url = f"{self.settings.DOWNLOAD_PUBLIC_URL.rstrip('/')}/download/exams/{identifier}.pdf"
        return ExamResult(identifier, status, len(questions), pdf_path, pdf_url, warnings)

    # -- Empaquetado de fuente ----------------------------------------------
    def build_source_zip(
        self,
        exam_id: str,
        *,
        format: str = "typst",
        downloads_dir: str | Path | None = None,
    ) -> Path:
        """Genera el ZIP con la fuente del examen (``main.typ``/``main.tex`` + assets).

        Valida ``exam_id`` y ``format`` para impedir traversal o lectura arbitraria.
        """
        if format not in _ALLOWED_FORMATS:
            raise ExamGenerationError(f"formato no soportado: {format!r}")
        if not _is_valid_exam_id(exam_id):
            raise ExamGenerationError("exam_id inválido")

        downloads = Path(downloads_dir or self.settings.DOWNLOADS_DIR)
        source_dir = downloads / "exams" / exam_id
        manifest_path = source_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ExamGenerationError(f"no existe un examen con id {exam_id!r}")

        manifest = json.loads(manifest_path.read_text())
        zip_path = downloads / f"fuente_examen_{exam_id}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            if format == "latex":
                archive.writestr("main.tex", self._manifest_to_latex(manifest))
            else:
                archive.write(source_dir / "main.typ", "main.typ")
            for asset in sorted((source_dir / "assets").glob("*")):
                if asset.is_file():
                    archive.write(asset, f"assets/{asset.name}")
            archive.writestr("README.txt", self._readme(format))
        return zip_path

    def _manifest_to_latex(self, manifest: dict[str, Any]) -> str:
        questions = [EnrichedQuestion.model_validate(q) for q in manifest["questions"]]
        return self.build_exam_latex(
            questions,
            title=manifest.get("title", "Examen de práctica"),
            figure_paths=manifest.get("figure_paths", {}),
        )

    @staticmethod
    def _readme(format: str) -> str:
        if format == "latex":
            return (
                "Fuente del examen en LaTeX.\n\n"
                "Compilar con: pdflatex main.tex\n\n"
                "Las figuras utilizadas están en la carpeta assets/.\n"
            )
        return (
            "Fuente del examen en Typst.\n\n"
            "Compilar con: typst compile main.typ output.pdf\n\n"
            "Las figuras utilizadas están en la carpeta assets/.\n"
        )
