"""Aplicación NiceGUI de revisión Human-in-the-Loop de Taygeta.

Modelo de estado por eventos (sin reruns): la página se construye una vez por
sesión y los manejadores mutan objetos Python que refrescan sus contenedores.
El flujo es:

1. subida de PDF/PNG/JPG al directorio ``UPLOADS_DIR``;
2. procesamiento con el pipeline de doble pase en background
   (``run.io_bound``, sin bloquear la interfaz);
3. revisión por pregunta: editor Markdown/TeX con previsualización, taxonomía
   con comboboxes de curso/tema (catálogo o escritura manual), figuras a la
   derecha con vistas «Figuras» / «Página original» y cropper de arrastre, y
   aprobación explícita antes de persistir en SQLite (``taygeta.db``).

Se ejecuta con ``python -m taygeta.ui.nice`` o con el entry point
``taygeta-ui``.
"""

from __future__ import annotations

import functools
import logging
import os
import re
from pathlib import Path
from typing import Any

from nicegui import app, run, ui

from taygeta.core.schema import (
    EnrichedQuestion,
    FigureReference,
    sanitize_math_delimiters,
)
from taygeta.core.settings import Settings, get_settings
from taygeta.domain.taxonomy import format_course_name
from taygeta.ingest.pipeline import IngestPipeline
from taygeta.server.db import Database, DatabaseError
from taygeta.server.object_storage import ObjectStorage
from taygeta.ui.nice import caption
from taygeta.ui.nice.question_panel import PanelDraft, QuestionPanel
from taygeta.ui.nice.state import QuestionState, SessionState, drop, register
from taygeta.ui.shared import (
    build_enriched_question,
    delete_asset,
    list_uploads,
    persist_approved_questions,
    public_s3_url,
    save_upload,
)

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"^[a-z0-9_]+$")


@functools.lru_cache(maxsize=1)
def _resources() -> tuple[Settings, Database, ObjectStorage, IngestPipeline]:
    """Recursos compartidos por todas las sesiones (análogo a ``cache_resource``)."""
    settings = get_settings()
    database = Database(settings=settings)
    storage = ObjectStorage(settings)
    pipeline = IngestPipeline(settings, storage=storage, database=database)
    return settings, database, storage, pipeline


@app.get("/health")
def _health() -> dict[str, str]:
    """Healthcheck simple para el orquestador (docker-compose)."""
    return {"status": "ok"}


def _refresh_docs(doc_select: ui.select, settings: Settings) -> None:
    """Actualiza el selector de documentos con el directorio de uploads."""
    names = [path.name for path in list_uploads(settings.UPLOADS_DIR)]
    current = doc_select.value
    doc_select.options = names
    if current not in names:
        doc_select.value = names[0] if names else None
    doc_select.update()


def _stat(label: str, value: str) -> None:
    """Mini-tarjeta de métrica para el resumen de la ingesta."""
    with ui.card().classes("w-28 items-center"):
        ui.label(value).classes("text-xl font-bold")
        ui.label(label).classes("text-xs text-gray-500")


async def _on_upload(
    e: Any, session: SessionState, doc_select: ui.select, settings: Settings
) -> None:
    """Guarda un archivo subido en ``UPLOADS_DIR`` y refresca el selector."""
    try:
        data = await e.file.read()
        path = save_upload(e.file.name, data, settings.UPLOADS_DIR)
    except Exception as exc:  # noqa: BLE001 - se informa al usuario, no se oculta
        ui.notify(f"No se pudo guardar el archivo: {exc}", type="negative")
        return
    _refresh_docs(doc_select, settings)
    ui.notify(f"Guardado en uploads: {path.name}", type="positive")


def _build_review(
    session: SessionState,
    settings: Settings,
    database: Database,
    storage: ObjectStorage,
    pipeline: IngestPipeline,
) -> list[QuestionPanel]:
    """Dibuja el resumen y los paneles de revisión; devuelve los paneles creados."""
    result = session.staging
    if result is None:
        return []
    panels: list[QuestionPanel] = []
    summary = result.summary()
    with ui.row().classes("w-full gap-4"):
        _stat("Estado", summary["status"])
        _stat("Total", str(summary["total"]))
        _stat("VALID", str(summary["valid"]))
        _stat("NEEDS_ATTENTION", str(summary["needs_attention"]))
    if summary["session_id"]:
        caption(f"Sesión de staging: `{summary['session_id']}`")
    if result.warnings:
        with ui.expansion("Advertencias de OCR"):
            ui.markdown("\n".join(f"- {warning}" for warning in result.warnings))
    caption(
        "La taxonomía, la dificultad y los prerrequisitos se infirieron automáticamente "
        "durante la estructuración; revísalos y edítalos antes de confirmar."
    )

    source_path = session.processed_path
    for index, staged in enumerate(result.questions):
        panels.append(
            QuestionPanel(
                ui.column().classes("w-full"),
                index=index,
                staged=staged,
                session=session,
                source_path=source_path,
                storage=storage,
                settings=settings,
                database=database,
                pipeline=pipeline,
            )
        )
    ui.separator()
    ui.button(
        "Confirmar y persistir preguntas aprobadas",
        color="primary",
        on_click=lambda: _confirm(session, panels, database, storage),
    )
    return panels


async def _on_process(
    session: SessionState,
    doc_select: ui.select,
    hint_input: ui.input,
    status_row: ui.row,
    status_label: ui.label,
    review: ui.column,
    settings: Settings,
    database: Database,
    storage: ObjectStorage,
    pipeline: IngestPipeline,
) -> None:
    """Procesa el documento seleccionado y (re)construye la revisión."""
    if session.processing:
        return
    if not doc_select.value:
        ui.notify("Selecciona un documento de la lista", type="warning")
        return
    path = Path(settings.UPLOADS_DIR) / str(doc_select.value)
    if str(path) != session.processed_path:
        session.staging = None
        session.questions = {}
        session.orphans = {}
    session.processing = True
    status_row.set_visibility(True)
    status_label.set_text(
        f"Procesando {path.name}: rasterización, OCR local, estructuración y validación…"
    )
    try:
        result = await run.io_bound(pipeline.ingest, path, hints=hint_input.value or None)
    except Exception as exc:  # noqa: BLE001 - se informa al usuario, no se oculta
        session.processing = False
        status_row.set_visibility(False)
        ui.notify(f"Error de ingesta: {exc}", type="negative")
        return
    session.processing = False
    status_row.set_visibility(False)
    if result is None:
        ui.notify("La ingesta fue cancelada.", type="warning")
        return
    session.processed_path = str(path)
    session.staging = result
    session.questions = {index: QuestionState() for index in range(len(result.questions))}
    review.clear()
    with review:
        _build_review(session, settings, database, storage, pipeline)
    ui.notify(f"Ingesta completada: {result.status}", type="positive")


def _confirm(
    session: SessionState,
    panels: list[QuestionPanel],
    database: Database,
    storage: ObjectStorage,
) -> None:
    """Recoge los paneles aprobados y persiste las preguntas en SQLite.

    Revalida los slugs de curso/tema y purga los assets huérfanos de las
    preguntas efectivamente confirmadas (nunca borra figuras de preguntas no
    confirmadas).
    """
    result = session.staging
    if result is None:
        return
    approved: list[tuple[EnrichedQuestion, list[str]]] = []
    _drafts: list[PanelDraft] = []
    for panel in panels:
        draft = panel.collect()
        if draft is None:
            continue
        _drafts.append(draft)
        if not draft.approved or draft.colgantes:
            continue
        approved_question = build_enriched_question(
            draft.base,
            course=draft.course,
            topic=draft.topic,
            subtopic=draft.subtopic,
            difficulty=draft.difficulty,
            prerequisites=[*draft.prerequisites, *draft.new_prereqs],
            source_metadata={"source_reference": result.source_reference},
            figures=[FigureReference.model_validate(figure) for figure in draft.effective_figures],
        )
        # Reflejar la edición manual (sanitiza delimitadores TeX no estándar).
        approved_question = approved_question.model_copy(
            update={
                "statement_raw": sanitize_math_delimiters(draft.statement),
                "question_type": draft.qtype,
                "choices": draft.choices,
            }
        )
        approved.append((approved_question, draft.new_prereqs))
    if not approved:
        ui.notify("No hay preguntas aprobadas para guardar.", type="warning")
        return
    invalid = [
        q for q, _ in approved if not _SLUG_RE.match(q.course) or not _SLUG_RE.match(q.topic)
    ]
    if invalid:
        ui.notify(
            "Curso y tema deben ser slugs válidos (minúsculas, dígitos y guion bajo).",
            type="negative",
        )
        return
    new_tags = sorted({slug for _, slugs in approved for slug in slugs})
    try:
        for question, _ in approved:
            database.upsert_course(question.course, format_course_name(question.course))
            database.upsert_topic(question.course, question.topic, question.subtopic)
        if new_tags:
            database.add_prerequisites(new_tags)
    except DatabaseError as exc:
        ui.notify(f"No se pudo actualizar la taxonomía: {exc}", type="negative")
        return
    confirm_result = persist_approved_questions(
        [question for question, _ in approved],
        database,
        storage,
        session_id=result.session_id,
    )
    _purge_orphans(session, set(confirm_result.saved), storage)
    if confirm_result.saved:
        ui.notify(
            f"Persistidas {len(confirm_result.saved)} pregunta(s) en taygeta.db.",
            type="positive",
        )
    for error in confirm_result.errors:
        ui.notify(error, type="negative")


def _purge_orphans(session: SessionState, confirmed_ids: set[str], storage: ObjectStorage) -> None:
    """Borra los assets huérfanos de las preguntas efectivamente confirmadas."""
    for question_id in confirmed_ids:
        for asset_path in session.orphans.get(str(question_id), set()):
            delete_asset(storage, asset_path)
    session.orphans = {}


@ui.page("/")
def _index_page() -> None:
    settings, database, storage, pipeline = _resources()
    session = register(ui.context.client.id)
    ui.page_title("Taygeta — Revisión HITL")

    with ui.column().classes("w-full max-w-7xl mx-auto gap-2 p-4"):
        ui.label("Taygeta — Ingesta y revisión Human-in-the-Loop").classes("text-2xl font-bold")

        # --- Estado y taxonomía ----------------------------------------
        with ui.expansion("Estado"):
            try:
                status = database.status()
                ui.markdown(
                    f"- **BD**: `{status['db_path']}` · integridad: {status['integrity']} · "
                    f"FTS5: {status['fts5_enabled']} · preguntas: {status['counts']['questions']}"
                )
            except DatabaseError as exc:
                ui.label(f"Base de datos no disponible: {exc}").classes("text-red-600")
            endpoint = public_s3_url(settings.S3_ENDPOINT_URL, settings)
            caption(f"S3: {settings.S3_BUCKET} @ {endpoint}")
            if not settings.llm_configured:
                ui.label(
                    "LLM remoto sin configurar: la validación semántica quedará en NEEDS_ATTENTION."
                ).classes("text-amber-600")
            with ui.expansion("Taxonomía"):
                ui.label("Cursos: " + ", ".join(c["slug"] for c in database.list_courses()))
                ui.label(
                    "Prerrequisitos: " + ", ".join(p["slug"] for p in database.list_prerequisites())
                )

        # --- 1. Subida ---------------------------------------------------
        ui.label("1. Subir documento").classes("text-lg font-semibold")
        ui.upload(
            label="PDF, PNG o JPG",
            auto_upload=True,
            on_upload=lambda e: _on_upload(e, session, doc_select, settings),
        )

        # --- 2. Procesar -------------------------------------------------
        ui.label("2. Procesar").classes("text-lg font-semibold")
        with ui.row().classes("w-full gap-4 items-center"):
            doc_select = ui.select([], label="Documento a procesar").classes("w-80")
            hint_input = ui.input("Contexto para el análisis (opcional)").classes("flex-1")
            process_btn = ui.button("Procesar documento", color="primary")
        status_row = ui.row().classes("items-center gap-2")
        with status_row:
            ui.spinner(size="lg")
            status_label = ui.label("").classes("text-gray-600")
        status_row.set_visibility(False)
        process_btn.on_click(
            lambda: _on_process(
                session,
                doc_select,
                hint_input,
                status_row,
                status_label,
                review,
                settings,
                database,
                storage,
                pipeline,
            )
        )
        _refresh_docs(doc_select, settings)

        # --- 3. Revisión y confirmación ----------------------------------
        ui.label("3. Revisión y confirmación").classes("text-lg font-semibold")
        review = ui.column().classes("w-full gap-4")
        if session.staging is not None:
            with review:
                _build_review(session, settings, database, storage, pipeline)


def main() -> None:
    """Arranca el servidor NiceGUI de la revisión HITL."""
    host = os.getenv("TAYGETA_UI_HOST", "0.0.0.0")
    port = int(os.getenv("TAYGETA_UI_PORT", "8501"))
    ui.run(
        host=host,
        port=port,
        title="Taygeta — Revisión HITL",
        show=False,
        reload=False,
        language="es",
    )


app.on_disconnect(lambda client: drop(client.id))
