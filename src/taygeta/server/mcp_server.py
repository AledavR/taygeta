"""Servidor MCP de Taygeta (sección 9).

Expone las herramientas FastMCP del flujo HITL y de recuperación RAG:

* ``stage_exam_upload`` (ingesta Headless con ``auto_commit``),
  ``refine_staged_question``, ``confirm_staged_upload`` (staging asistido por chat);
* ``search_questions`` y ``list_taxonomy`` (recuperación RAG y catálogo);
* ``generate_practice_exam`` y ``get_exam_source`` (generación y compilación).

Además monta la ruta HTTP ``/download/{filename}`` (y una ``/assets/{asset_id}``
controlada) y un limpiador en segundo plano con TTL de 60 minutos para los
archivos temporales de ``/app/downloads``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response

from taygeta.core.schema import (
    Choice,
    EnrichedQuestion,
    QuestionRaw,
    TranscriptionIntermediate,
    sanitize_math_delimiters,
    utc_now_iso,
)
from taygeta.core.settings import Settings, get_settings
from taygeta.core.urls import build_public_url, build_relative_path, split_asset_reference
from taygeta.domain.taxonomy import format_course_name
from taygeta.ingest.pipeline import IngestPipeline, StagingResult
from taygeta.ingest.validator import STATUS_NEEDS_ATTENTION, STATUS_VALID
from taygeta.server.db import Database
from taygeta.server.deduplication import DeduplicationService
from taygeta.server.exam_generator import ExamGenerationError, ExamGenerator
from taygeta.server.object_storage import ObjectStorage, ObjectStorageError

logger = logging.getLogger(__name__)

__all__ = ["mcp", "main", "start_download_cleaner"]

#: Campos editables de una pregunta que admiten overrides manuales.
_ALLOWED_OVERRIDES = frozenset({"statement_raw", "question_type", "choices", "notation_notes"})

#: Campos curriculares que se adjuntan al payload de staging para su confirmación.
_CURRICULAR_OVERRIDES = frozenset({"course", "topic", "subtopic", "difficulty", "prerequisites"})

#: Límite máximo de resultados de búsqueda para proteger el servicio.
_MAX_SEARCH_LIMIT = 100

#: Caché temporal (en memoria) de las sesiones headless pendientes de revisión.
_headless_cache: dict[str, dict[str, Any]] = {}

_settings = get_settings()
mcp = FastMCP("taygeta", host=_settings.MCP_HOST, port=_settings.MCP_PORT)


def _resources() -> tuple[Settings, Database, ObjectStorage, IngestPipeline, ExamGenerator]:
    settings = get_settings()
    database = Database(settings=settings)
    storage = ObjectStorage(settings)
    pipeline = IngestPipeline(settings, storage=storage, database=database)
    exams = ExamGenerator(settings, database=database, storage=storage)
    return settings, database, storage, pipeline, exams


# ---------------------------------------------------------------------------
# Utilidades de seguridad y formato
# ---------------------------------------------------------------------------


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _public_question(question: EnrichedQuestion, settings: Settings) -> dict[str, Any]:
    """Serializa una pregunta resolviendo las rutas de figuras a URLs públicas.

    Antes de enviar la información al frontend o al cliente MCP, cada ruta
    relativa de figura (``/buckets/{bucket}/{key}``) se convierte en su URL
    completa concatenando ``S3_PUBLIC_BASE_URL``.
    """
    data = question.model_dump()
    for figure in data.get("figures", []):
        reference = figure.get("asset_path")
        if reference:
            bucket, key = split_asset_reference(reference, default_bucket=settings.S3_BUCKET)
            figure["asset_path"] = build_public_url(build_relative_path(bucket, key), settings)
    return data


def _resolve_upload_path(file_path: str, settings: Settings) -> Path:
    """Valida que ``file_path`` esté dentro de ``UPLOADS_DIR`` y sea un archivo."""
    uploads = Path(settings.UPLOADS_DIR).expanduser().resolve()
    candidate = Path(file_path)
    if not candidate.is_absolute():
        candidate = uploads / candidate
    candidate = candidate.resolve()
    if candidate != uploads and not candidate.is_relative_to(uploads):
        raise ValueError("la ruta está fuera del directorio de uploads")
    if not candidate.is_file():
        raise ValueError(f"el archivo no existe: {candidate}")
    return candidate


def _resolve_download_path(filename: str, settings: Settings) -> Path:
    """Valida un nombre de archivo de descarga para impedir traversal."""
    downloads = Path(settings.DOWNLOADS_DIR).expanduser().resolve()
    if not filename or filename.startswith("/") or ".." in Path(filename).parts:
        raise ValueError("nombre de archivo inválido")
    candidate = (downloads / filename).resolve()
    if not candidate.is_relative_to(downloads):
        raise ValueError("la ruta de descarga está fuera del directorio permitido")
    if not candidate.is_file():
        raise ValueError("archivo no encontrado")
    return candidate


def _apply_overrides(question: QuestionRaw, overrides: dict[str, Any] | None) -> QuestionRaw:
    """Aplica únicamente los overrides manuales permitidos sobre la pregunta.

    ``model_copy`` no ejecuta los validadores de Pydantic, por lo que el enunciado
    y las alternativas manuales se sanitizan aquí explícitamente para no
    reintroducir delimitadores TeX no estándar (``\\[...\\]``, ``\\(...\\)``).
    """
    if not overrides:
        return question
    updates: dict[str, Any] = {
        key: value for key, value in overrides.items() if key in _ALLOWED_OVERRIDES
    }
    if isinstance(updates.get("statement_raw"), str):
        updates["statement_raw"] = sanitize_math_delimiters(updates["statement_raw"])
    if "choices" in updates and updates["choices"] is not None:
        updates["choices"] = [Choice.model_validate(choice) for choice in updates["choices"]]
    return question.model_copy(update=updates)


# ---------------------------------------------------------------------------
# Herramientas MCP — staging asistido por chat
# ---------------------------------------------------------------------------


@mcp.tool()
def stage_exam_upload(file_path: str, auto_commit: bool = True) -> str:
    """Procesa un archivo de /app/uploads con el flujo Headless.

    Ejecuta el pipeline de doble pase y las dos capas de validación. Si todas las
    preguntas son ``VALID`` y ``auto_commit`` está activo, las guarda directamente
    en la base de datos y devuelve éxito. Si hay incidencias (``NEEDS_ATTENTION``,
    ``ERROR`` o ``DUPLICATE``), genera una sesión, la guarda en caché temporal y
    devuelve la URL de revisión en la UI sin escribir en la base de datos.
    """
    try:
        settings, database, storage, pipeline, _exams = _resources()
        path = _resolve_upload_path(file_path, settings)
        result = pipeline.ingest(path, persist_staging=False)
    except Exception as exc:  # noqa: BLE001 - se informa al chat, no se oculta
        return _json({"status": "error", "message": str(exc)})

    summary = result.summary()
    problems = [
        {
            "id": item.question.id if item.question is not None else None,
            "status": item.status,
            "reasons": item.reasons,
        }
        for item in result.questions
        if item.status != STATUS_VALID or item.question is None
    ]

    if auto_commit and result.status == STATUS_VALID and not problems:
        saved, save_problems = _save_headless(
            result, database, storage, DeduplicationService(database)
        )
        if not save_problems:
            return _json(
                {
                    "status": "ok",
                    "auto_commit": True,
                    "saved": saved,
                    "total": summary["total"],
                    "message": (
                        f"{len(saved)} pregunta(s) guardadas automáticamente en la base de datos."
                    ),
                }
            )
        problems = save_problems

    session_id = str(uuid.uuid4())
    _headless_cache[session_id] = _staging_payload(result)
    review_url = f"{settings.STREAMLIT_PUBLIC_URL.rstrip('/')}/?session_id={session_id}"
    status = STATUS_NEEDS_ATTENTION if problems else result.status
    return _json(
        {
            "status": status,
            "session_id": session_id,
            "review_url": review_url,
            "total": summary["total"],
            "valid": summary["valid"],
            "needs_attention": summary["needs_attention"],
            "problems": problems,
            "warnings": result.warnings,
            "message": "hay incidencias; revisa la sesión en la UI: " + review_url,
        }
    )


@mcp.tool()
def refine_staged_question(
    session_id: str,
    q_index: int,
    user_hint: str,
    manual_overrides: dict[str, Any] | None = None,
) -> str:
    """Reanaliza una pregunta de staging con contexto correctivo.

    Aplica los overrides manuales (enunciado, tipo, alternativas y clasificación
    curricular) y los conserva explícitamente. No resuelve el ejercicio.
    """
    try:
        _settings, database, _storage, pipeline, _exams = _resources()
        session = database.get_staging_session(session_id)
        if session is None:
            return _json({"status": "error", "message": f"sesión no encontrada: {session_id}"})
        data = session["data"]
        questions = data.get("questions", [])
        if q_index < 0 or q_index >= len(questions):
            return _json({"status": "error", "message": f"índice fuera de rango: {q_index}"})

        overrides = manual_overrides or {}
        raw_overrides = {
            key: value for key, value in overrides.items() if key in _ALLOWED_OVERRIDES
        }
        curricular = {
            key: value for key, value in overrides.items() if key in _CURRICULAR_OVERRIDES
        }

        intermediate = _intermediate_from_session(data)
        current_payload = questions[q_index].get("question")
        if current_payload is None:
            return _json(
                {
                    "status": "error",
                    "message": f"la pregunta {q_index} no tiene enunciado que reanalizar",
                }
            )
        current = EnrichedQuestion.model_validate(current_payload)
        refined = pipeline.refine_question(intermediate, current, hints=user_hint or None)
        if refined.question is not None:
            refined.question = _apply_overrides(refined.question, raw_overrides)
            refined = pipeline.validate_question(
                refined.question, max_pages=len(intermediate.pages) or None
            )
            payload = refined.question.model_dump()
            payload.update(curricular)
            payload.setdefault("created_at", utc_now_iso())
        else:
            payload = None
        questions[q_index] = {
            "status": refined.status,
            "reasons": refined.reasons,
            "question": payload,
        }
        data["questions"] = questions
        database.update_staging_session(session_id, data=data)
        return _json(
            {
                "status": refined.status,
                "q_index": q_index,
                "reasons": refined.reasons,
                "question": payload,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _json({"status": "error", "message": str(exc)})


@mcp.tool()
def confirm_staged_upload(session_id: str) -> str:
    """Confirma una sesión de staging y persiste las preguntas aprobadas.

    Es idempotente: una sesión ya confirmada se informa sin reinsertar.
    """
    try:
        _settings, database, storage, _pipeline, _exams = _resources()
        dedup = DeduplicationService(database)
        session = database.get_staging_session(session_id)
        if session is None:
            return _json({"status": "error", "message": f"sesión no encontrada: {session_id}"})
        if session["status"] == "CONFIRMED":
            return _json(
                {"status": "ok", "message": f"la sesión {session_id} ya estaba confirmada"}
            )

        saved: list[str] = []
        skipped: list[str] = []
        errors: list[str] = []
        for item in session["data"].get("questions", []):
            if item.get("status") != STATUS_VALID:
                skipped.append(f"{item.get('question', {}).get('id', '?')} (no VALID)")
                continue
            payload = item.get("question")
            if not payload:
                continue
            question = _enrich(payload, storage)
            if question is None:
                skipped.append(f"{payload.get('id', '?')} (sin clasificación curricular)")
                continue
            try:
                database.add_prerequisites(list(question.prerequisites))
                database.upsert_course(question.course, format_course_name(question.course))
                database.upsert_topic(question.course, question.topic, question.subtopic)
                database.upsert_question(question, embedding=dedup.embed_question(question))
                saved.append(question.id)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{payload.get('id', '?')}: {exc}")

        if saved:
            database.update_staging_session(session_id, status="CONFIRMED")
        return _json(
            {
                "status": "ok" if saved else "error",
                "saved": saved,
                "skipped": skipped,
                "errors": errors,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _json({"status": "error", "message": str(exc)})


# ---------------------------------------------------------------------------
# Herramientas MCP — consultas y RAG
# ---------------------------------------------------------------------------


@mcp.tool()
def search_questions(
    course: str = "",
    topic: str = "",
    difficulty: int = 0,
    limit: int = 20,
) -> str:
    """Recupera preguntas reales por curso, tema y dificultad (sin soluciones)."""
    try:
        settings, database, _storage, _pipeline, _exams = _resources()
        limit = max(1, min(limit, _MAX_SEARCH_LIMIT))
        questions = database.search_questions(
            course=course or None,
            topic=topic or None,
            difficulty=difficulty if difficulty >= 1 else None,
            limit=limit,
        )
        return _json(
            {
                "count": len(questions),
                "questions": [_public_question(q, settings) for q in questions],
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _json({"status": "error", "message": str(exc)})


@mcp.tool()
def list_taxonomy() -> str:
    """Consulta el catálogo de cursos, temas, subtemas y prerrequisitos."""
    try:
        _settings, database, _storage, _pipeline, _exams = _resources()
        return _json(database.list_taxonomy())
    except Exception as exc:  # noqa: BLE001
        return _json({"status": "error", "message": str(exc)})


# ---------------------------------------------------------------------------
# Herramientas MCP — generación y compilación
# ---------------------------------------------------------------------------


@mcp.tool()
def generate_practice_exam(
    course: str = "",
    topics: list[str] | None = None,
    difficulty: int = 0,
    num_questions: int = 10,
) -> str:
    """Genera y compila un examen de práctica en PDF con Typst."""
    try:
        _settings, _database, _storage, _pipeline, exams = _resources()
        result = exams.generate_exam(
            course=course or None,
            topics=topics or None,
            difficulty=difficulty if difficulty >= 1 else None,
            num_questions=num_questions,
        )
        return _json(result.summary())
    except Exception as exc:  # noqa: BLE001
        return _json({"status": "error", "message": str(exc)})


@mcp.tool()
def get_exam_source(exam_id: str, format: str = "typst") -> str:
    """Genera y devuelve la URL de un ZIP con la fuente del examen."""
    try:
        settings, _database, _storage, _pipeline, exams = _resources()
        zip_path = exams.build_source_zip(exam_id, format=format)
        url = f"{settings.DOWNLOAD_PUBLIC_URL.rstrip('/')}/download/{zip_path.name}"
        return _json({"exam_id": exam_id, "format": format, "zip_url": url, "path": str(zip_path)})
    except (ExamGenerationError, ValueError) as exc:
        return _json({"status": "error", "message": str(exc)})
    except Exception as exc:  # noqa: BLE001
        return _json({"status": "error", "message": str(exc)})


# ---------------------------------------------------------------------------
# Helpers internos de staging
# ---------------------------------------------------------------------------


def _intermediate_from_session(data: dict[str, Any]) -> TranscriptionIntermediate:
    intermediate = data.get("intermediate")
    if intermediate is not None:
        return TranscriptionIntermediate.model_validate(intermediate)
    questions = data.get("questions", [])
    first = questions[0].get("question") if questions else None
    return TranscriptionIntermediate(
        source_reference=data.get("source_reference", "staging"),
        raw_markdown=first.get("statement_raw", "") if first else "",
        pages=[],
        figures=[],
        warnings=[],
    )


def _staging_payload(result: StagingResult) -> dict[str, Any]:
    """Serializa un resultado de staging para la caché temporal headless."""
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
                "question": item.question.model_dump() if item.question is not None else None,
            }
            for item in result.questions
        ],
    }


def _save_headless(
    result: StagingResult,
    database: Database,
    storage: ObjectStorage,
    dedup: DeduplicationService,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Persiste automáticamente las preguntas VALID de una ingesta headless.

    Usa :class:`DeduplicationService` para detectar duplicados semánticos (por
    embeddings y, opcionalmente, juez LLM) y captura cualquier excepción como
    ``ERROR``. Devuelve los ids guardados y la lista de incidencias.
    """
    saved: list[str] = []
    problems: list[dict[str, Any]] = []
    for item in result.questions:
        question = item.question
        if item.status != STATUS_VALID or question is None:
            problems.append(
                {
                    "id": question.id if question is not None else None,
                    "status": item.status,
                    "reasons": item.reasons,
                }
            )
            continue
        try:
            enriched = _enrich(question.model_dump(), storage)
            if enriched is None:
                problems.append(
                    {
                        "id": question.id,
                        "status": "ERROR",
                        "reasons": ["sin clasificación curricular"],
                    }
                )
                continue
            database.add_prerequisites(list(enriched.prerequisites))
            database.upsert_course(enriched.course, format_course_name(enriched.course))
            database.upsert_topic(enriched.course, enriched.topic, enriched.subtopic)
            outcome = dedup.ingest(enriched)
        except Exception as exc:  # noqa: BLE001
            problems.append({"id": question.id, "status": "ERROR", "reasons": [str(exc)]})
            continue
        if outcome.status == "DUPLICATE":
            problems.append(
                {
                    "id": question.id,
                    "status": "DUPLICATE",
                    "reasons": [outcome.reason or "pregunta duplicada"],
                }
            )
        else:
            saved.append(question.id)
    return saved, problems


def _enrich(payload: dict[str, Any], storage: ObjectStorage) -> EnrichedQuestion | None:
    """Construye una ``EnrichedQuestion`` a partir del payload de staging.

    Requiere clasificación curricular (course/topic/difficulty) y verifica que
    las figuras estén en el bucket persistente antes de persistir.
    """
    payload = dict(payload)
    for key in ("course", "topic", "difficulty"):
        if key not in payload:
            return None
    payload.setdefault("created_at", utc_now_iso())
    payload.setdefault("source_metadata", {})
    question = EnrichedQuestion.model_validate(payload)
    for figure in question.figures:
        if not figure.asset_path:
            raise ValueError(f"la figura {figure.id} no tiene identificador persistente")
        bucket, key = split_asset_reference(figure.asset_path)
        if bucket:
            if not storage.exists(key, bucket=bucket):
                raise ValueError(f"la figura {figure.id} no está en el bucket")
        elif not storage.exists(figure.asset_path):
            raise ValueError(f"la figura {figure.id} no está en el bucket")
    return question


# ---------------------------------------------------------------------------
# Rutas HTTP y limpiador TTL
# ---------------------------------------------------------------------------


@mcp.custom_route("/health", methods=["GET"])
async def health(_request: Request) -> Response:
    """Endpoint de salud para healthchecks de Docker Compose."""
    return JSONResponse({"status": "ok"})


@mcp.custom_route("/download/{filename:path}", methods=["GET"])
async def download_file(request: Request) -> Response:
    """Sirve un archivo temporal del directorio de descargas."""
    settings = get_settings()
    filename = request.path_params.get("filename", "")
    try:
        path = _resolve_download_path(filename, settings)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    media_type = "application/pdf" if path.suffix == ".pdf" else "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name)


@mcp.custom_route("/assets/{asset_id:path}", methods=["GET"])
async def asset_proxy(request: Request) -> Response:
    """Devuelve una figura desde SeaweedFS por su identificador controlado.

    No es un proxy S3 arbitrario: el identificador debe ser una clave o URI
    s3:// válida, y se valida contra el almacenamiento configurado.
    """
    settings = get_settings()
    asset_id = request.path_params.get("asset_id", "")
    storage = ObjectStorage(settings)
    try:
        bucket, key = storage.resolve(asset_id)
        data = storage.get_bytes(key, bucket=bucket)
    except ObjectStorageError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    content_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
    }.get(Path(key).suffix.lower(), "application/octet-stream")
    return Response(content=data, media_type=content_type)


def _clean_downloads(settings: Settings) -> int:
    """Elimina archivos temporales de ``DOWNLOADS_DIR`` con más de ``TTL`` segundos."""
    downloads = Path(settings.DOWNLOADS_DIR).expanduser()
    if not downloads.is_dir():
        return 0
    cutoff = time.time() - settings.DOWNLOADS_TTL_SECONDS
    removed = 0
    for path in downloads.rglob("*"):
        if path.is_file():
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError as exc:  # noqa: BLE001
                logger.warning("no se pudo eliminar %s: %s", path, exc)
    return removed


def start_download_cleaner(
    settings: Settings | None = None, *, interval_seconds: int = 300
) -> threading.Thread:
    """Inicia un thread en segundo plano que limpia descargas expiradas.

    Solo toca archivos de ``DOWNLOADS_DIR``: nunca la base de datos, las cargas
    originales ni los objetos persistentes del bucket.
    """
    resolved = settings or get_settings()

    def _loop() -> None:
        while True:
            try:
                removed = _clean_downloads(resolved)
                if removed:
                    logger.info("limpiador de descargas eliminó %d archivo(s)", removed)
            except Exception as exc:  # noqa: BLE001
                logger.warning("error en el limpiador de descargas: %s", exc)
            time.sleep(interval_seconds)

    thread = threading.Thread(target=_loop, name="taygeta-download-cleaner", daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------


def main() -> None:
    """Arranca el servidor MCP con transporte HTTP y el limpiador TTL."""
    settings = get_settings()
    start_download_cleaner(settings)
    logger.info("servidor MCP de Taygeta escuchando en %s:%s", settings.MCP_HOST, settings.MCP_PORT)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
