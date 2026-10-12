"""Interfaz Human-in-the-Loop de Taygeta (sección 8).

Aplicación Streamlit para revisar la ingesta híbrida:

* Subida de PDF/PNG/JPG al directorio ``UPLOADS_DIR`` y procesamiento con el
  pipeline de doble pase, mostrando el estado.
* Panel de revisión con visualización **en paralelo** del enunciado Markdown y
  de la imagen original (recorte de figura desde SeaweedFS o página rasterizada).
* Estados ``VALID`` / ``NEEDS_ATTENTION`` con explicación técnica.
* Campo de reintento con contexto correctivo que reanaliza el enunciado.
* Edición manual del Markdown y de los campos, slider de dificultad, selectores
  de curso/tema/subtema y multiselect de prerrequisitos con alta dinámica.
* Botón de confirmación que persiste las preguntas **aprobadas** en ``taygeta.db``
  y garantiza que las figuras estén en el bucket persistente de SeaweedFS.

La corrección manual prevalece sobre la transcripción automática. Ninguna
pregunta se guarda en la base definitiva sin la confirmación explícita del
usuario, y un fallo de OCR o de validación nunca se oculta marcando la pregunta
como válida.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import streamlit as st

from taygeta.core.schema import (
    Choice,
    EnrichedQuestion,
    QuestionRaw,
    TranscriptionIntermediate,
    utc_now_iso,
)
from taygeta.core.settings import Settings, get_settings
from taygeta.ingest.image_extractor import (
    is_supported_source,
    open_document,
    rasterize_page,
)
from taygeta.ingest.pipeline import IngestPipeline, StagedQuestion, StagingResult
from taygeta.ingest.validator import STATUS_NEEDS_ATTENTION, STATUS_VALID
from taygeta.server.db import Database, DatabaseError
from taygeta.server.object_storage import ObjectStorage, ObjectStorageError, parse_s3_uri

logger = logging.getLogger(__name__)

__all__ = [
    "ConfirmResult",
    "build_enriched_question",
    "figure_bytes",
    "list_uploads",
    "main",
    "page_image_bytes",
    "persist_approved_questions",
    "sanitize_upload_name",
    "save_upload",
]

#: Extensiones aceptadas por el cargador de archivos.
UPLOAD_EXTENSIONS = ["pdf", "png", "jpg", "jpeg"]

_NEW_COURSE = "<nuevo curso>"
_NEW_TOPIC = "<nuevo tema>"
_NEW_SUBTOPIC = "<nuevo subtema>"
_NO_SUBTOPIC = "(ninguno)"
_SLUG_RE = re.compile(r"^[a-z0-9_]+$")


# ---------------------------------------------------------------------------
# Utilidades de archivos y figuras (testables sin Streamlit)
# ---------------------------------------------------------------------------


def sanitize_upload_name(name: str) -> str:
    """Sanea el nombre de un archivo subido, evitando traversal y caracteres raros."""
    base = Path(name).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", base).lstrip(".")
    return cleaned or "upload"


def save_upload(name: str, data: bytes, uploads_dir: str | Path) -> Path:
    """Guarda los bytes subidos en ``uploads_dir`` y devuelve la ruta final."""
    directory = Path(uploads_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / sanitize_upload_name(name)
    if destination.exists():
        destination = directory / f"{destination.stem}_{uuid.uuid4().hex[:8]}{destination.suffix}"
    destination.write_bytes(data)
    return destination


def list_uploads(uploads_dir: str | Path) -> list[Path]:
    """Lista los documentos admitidos presentes en ``uploads_dir``."""
    directory = Path(uploads_dir)
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and is_supported_source(path)),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def figure_bytes(storage: ObjectStorage | None, asset_path: str | None) -> bytes | None:
    """Recupera los bytes de una figura desde SeaweedFS a partir de su referencia.

    Acepta una URI ``s3://bucket/key`` o una clave de objeto. Devuelve ``None``
    si la referencia está vacía o el objeto no se puede recuperar.
    """
    if storage is None or not asset_path:
        return None
    try:
        if asset_path.startswith("s3://"):
            bucket, key = parse_s3_uri(asset_path)
            return storage.get_bytes(key, bucket=bucket)
        return storage.get_bytes(asset_path)
    except ObjectStorageError as exc:
        logger.warning("no se pudo recuperar la figura %s: %s", asset_path, exc)
        return None


def page_image_bytes(source_path: str | Path, page_index: int, *, dpi: int = 150) -> bytes | None:
    """Rasteriza una página del documento origen para mostrarla en la revisión."""
    try:
        with open_document(source_path) as document:
            if page_index < 0 or page_index >= document.page_count:
                return None
            page = document.load_page(page_index)
            is_pdf = bool(getattr(document, "is_pdf", False))
            return rasterize_page(page, dpi=dpi, is_pdf=is_pdf).png_bytes
    except Exception as exc:  # noqa: BLE001 - la vista previa nunca debe romper la UI
        logger.warning("no se pudo rasterizar la página %s: %s", page_index, exc)
        return None


# ---------------------------------------------------------------------------
# Construcción y persistencia de preguntas aprobadas
# ---------------------------------------------------------------------------


def build_enriched_question(
    base: QuestionRaw,
    *,
    course: str,
    topic: str,
    subtopic: str | None,
    difficulty: int,
    prerequisites: Sequence[str],
    source_metadata: dict[str, Any] | None = None,
) -> EnrichedQuestion:
    """Compone una :class:`EnrichedQuestion` a partir de los campos revisados.

    Los valores editados manualmente prevalecen sobre la transcripción
    automática, ya que se toman directamente del formulario.
    """
    return EnrichedQuestion(
        id=base.id,
        statement_raw=base.statement_raw,
        question_type=base.question_type,
        choices=base.choices,
        figures=base.figures,
        notation_notes=base.notation_notes,
        course=course,
        topic=topic,
        subtopic=subtopic or None,
        difficulty=difficulty,
        prerequisites=list(prerequisites),
        source_metadata=source_metadata or {},
        created_at=utc_now_iso(),
    )


@dataclass(slots=True)
class ConfirmResult:
    """Resultado de la confirmación y persistencia de preguntas."""

    saved: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _missing_figure_assets(question: EnrichedQuestion, storage: ObjectStorage | None) -> list[str]:
    missing: list[str] = []
    for figure in question.figures:
        if not figure.asset_path:
            missing.append(f"{figure.id} (sin asset_path)")
            continue
        if storage is None:
            continue
        try:
            if figure.asset_path.startswith("s3://"):
                bucket, key = parse_s3_uri(figure.asset_path)
                if not storage.exists(key, bucket=bucket):
                    missing.append(figure.asset_path)
            elif not storage.exists(figure.asset_path):
                missing.append(figure.asset_path)
        except ObjectStorageError:
            missing.append(figure.asset_path)
    return missing


def persist_approved_questions(
    questions: Sequence[EnrichedQuestion],
    database: Database,
    storage: ObjectStorage | None = None,
    *,
    session_id: str | None = None,
) -> ConfirmResult:
    """Persiste las preguntas aprobadas tras verificar sus figuras en el bucket.

    Usa ``upsert_question`` para que la confirmación sea idempotente. Solo marca
    la sesión de staging como ``CONFIRMED`` cuando ha guardado alguna pregunta.
    """
    result = ConfirmResult()
    for question in questions:
        try:
            missing = _missing_figure_assets(question, storage)
            if missing:
                result.errors.append(
                    f"{question.id}: figuras no presentes en el bucket: {', '.join(missing)}"
                )
                continue
            database.add_prerequisites(list(question.prerequisites))
            database.upsert_question(question)
            result.saved.append(question.id)
        except Exception as exc:  # noqa: BLE001 - se informa al usuario, no se oculta
            result.errors.append(f"{question.id}: {exc}")

    if session_id and result.saved:
        try:
            database.update_staging_session(session_id, status="CONFIRMED")
        except DatabaseError as exc:
            result.errors.append(f"no se pudo marcar la sesión como confirmada: {exc}")
    return result


# ---------------------------------------------------------------------------
# Recursos cacheados
# ---------------------------------------------------------------------------


@st.cache_resource
def _resources() -> tuple[Settings, Database, ObjectStorage, IngestPipeline]:
    settings = get_settings()
    database = Database(settings.DB_PATH, settings=settings)
    storage = ObjectStorage(settings)
    pipeline = IngestPipeline(settings, storage=storage, database=database)
    return settings, database, storage, pipeline


# ---------------------------------------------------------------------------
# Renderizado
# ---------------------------------------------------------------------------


def _render_reference_image(
    source_path: str | None,
    staged: StagedQuestion,
    storage: ObjectStorage | None,
    container: Any,
) -> None:
    """Muestra el recorte de figura o la página original junto al enunciado."""
    question = staged.question
    figures = question.figures if question is not None else []
    shown = False
    for figure in figures:
        data = figure_bytes(storage, figure.asset_path)
        if data:
            container.image(data, caption=f"{figure.id} — recorte (página {figure.page})")
            shown = True
        elif figure.asset_path:
            try:
                container.image(
                    storage.display_url(figure.asset_path) if storage else figure.asset_path,
                    caption=f"{figure.id} — recorte (página {figure.page})",
                )
                shown = True
            except ObjectStorageError:
                shown = False
    if shown or source_path is None:
        return
    pages = [0]
    if question is not None and question.figures:
        pages = sorted({figure.page for figure in question.figures})
    for page_index in pages:
        data = page_image_bytes(source_path, page_index)
        if data:
            container.image(data, caption=f"Página original {page_index}")


def _render_taxonomy_controls(
    index: int, database: Database, staged: StagedQuestion
) -> tuple[str, str, str | None, int, list[str], list[str]]:
    """Dibuja los selectores de taxonomía y devuelve los valores elegidos."""
    base = staged.question
    courses = sorted({course["slug"] for course in database.list_courses()})
    course_choice = st.selectbox("Curso", [*courses, _NEW_COURSE], key=f"course_{index}")
    course = course_choice
    if course_choice == _NEW_COURSE:
        course = st.text_input("Slug del nuevo curso", key=f"new_course_{index}").strip()

    topics = sorted(
        {topic["topic_slug"] for topic in database.list_topics(course)} if course else set()
    )
    topic_choice = st.selectbox("Tema", [*topics, _NEW_TOPIC], key=f"topic_{index}")
    topic = topic_choice
    if topic_choice == _NEW_TOPIC:
        topic = st.text_input("Slug del nuevo tema", key=f"new_topic_{index}").strip()

    subtopics = sorted(
        {
            topic_row["subtopic_slug"]
            for topic_row in database.list_topics(course)
            if topic_row["topic_slug"] == topic and topic_row["subtopic_slug"]
        }
        if course
        else set()
    )
    subtopic_options = [_NO_SUBTOPIC, *subtopics, _NEW_SUBTOPIC]
    subtopic_choice = st.selectbox("Subtema", subtopic_options, key=f"subtopic_{index}")
    if subtopic_choice == _NEW_SUBTOPIC:
        subtopic: str | None = st.text_input(
            "Slug del nuevo subtema", key=f"new_subtopic_{index}"
        ).strip()
    elif subtopic_choice == _NO_SUBTOPIC:
        subtopic = None
    else:
        subtopic = subtopic_choice

    difficulty = st.slider("Dificultad", min_value=1, max_value=5, value=3, key=f"diff_{index}")

    catalog = sorted({prereq["slug"] for prereq in database.list_prerequisites()})
    default_prereqs = list(base.prerequisites) if base is not None else []
    prerequisites = st.multiselect(
        "Prerrequisitos", catalog, default=default_prereqs, key=f"prereq_{index}"
    )
    new_prereqs_raw = st.text_input(
        "Nuevos prerrequisitos (separados por coma)", key=f"new_prereq_{index}"
    )
    new_prereqs = [slug.strip() for slug in new_prereqs_raw.split(",") if slug.strip()]
    return course, topic, subtopic, difficulty, prerequisites, new_prereqs


def _render_review(
    result: StagingResult,
    settings: Settings,
    database: Database,
    storage: ObjectStorage,
    source_path: str | None,
) -> None:
    summary = result.summary()
    metrics = st.columns(4)
    metrics[0].metric("Estado", summary["status"])
    metrics[1].metric("Total", summary["total"])
    metrics[2].metric("VALID", summary["valid"])
    metrics[3].metric("NEEDS_ATTENTION", summary["needs_attention"])
    if summary["session_id"]:
        st.caption(f"Sesión de staging: `{summary['session_id']}`")
    if result.warnings:
        joined = "\n".join(f"- {warning}" for warning in result.warnings)
        st.warning("Advertencias de OCR:\n" + joined)

    approved: list[tuple[EnrichedQuestion, list[str]]] = []

    for index, staged in enumerate(result.questions):
        label = f"Pregunta {index + 1} — {staged.status}"
        with st.expander(label, expanded=staged.status == STATUS_NEEDS_ATTENTION):
            if staged.status == STATUS_VALID:
                st.success("Validación superada (VALID). Revísala y apruébala para guardarla.")
            else:
                st.error("NEEDS_ATTENTION: " + "; ".join(staged.reasons))
            if staged.logical is not None and not staged.logical.is_valid:
                st.error(f"Motivo de rechazo semántico: {staged.logical.rejection_reason}")

            if staged.question is None:
                st.info("No hay enunciado estructurado para editar en esta pregunta.")
                continue

            base = staged.question
            left, right = st.columns(2)
            with left:
                new_statement = st.text_area(
                    "Enunciado (Markdown/TeX)", value=base.statement_raw, key=f"statement_{index}"
                )
                question_types = ["multiple_choice", "open_ended", "true_false", "numerical"]
                qtype = st.selectbox(
                    "Tipo de pregunta",
                    question_types,
                    index=question_types.index(base.question_type),
                    key=f"qtype_{index}",
                )
                new_choices: list[Choice] | None = None
                if qtype == "multiple_choice":
                    choices: list[Choice] = []
                    for choice_index, choice in enumerate(base.choices or []):
                        choice_id = st.text_input(
                            f"Alternativa {choice_index + 1} — id",
                            value=choice.id,
                            key=f"choice_id_{index}_{choice_index}",
                        )
                        choice_text = st.text_area(
                            f"Alternativa {choice_index + 1} — contenido",
                            value=choice.content_raw,
                            key=f"choice_text_{index}_{choice_index}",
                        )
                        choices.append(Choice(id=choice_id, content_raw=choice_text))
                    new_choices = choices
                if base.notation_notes:
                    st.info("Notas de notación: " + "; ".join(base.notation_notes))
            with right:
                st.caption("Enunciado renderizado")
                st.markdown(new_statement)
                _render_reference_image(source_path, staged, storage, st)

            hint = st.text_input(
                "Contexto correctivo (p. ej. «La condición ilegible es g(x) = ...»)",
                key=f"hint_{index}",
            )
            if st.button("Reanalizar enunciado", key=f"refine_{index}"):
                intermediate = result.intermediate
                if intermediate is None:
                    intermediate = TranscriptionIntermediate(
                        source_reference=result.source_reference,
                        raw_markdown=base.statement_raw,
                        pages=[],
                        figures=base.figures,
                        warnings=[],
                    )
                try:
                    refined = st.session_state["_pipeline"].refine_question(
                        intermediate, index, hints=hint or None
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"No se pudo reanalizar: {exc}")
                else:
                    result.questions[index] = refined
                    st.session_state["staging_result"] = result
                    st.rerun()

            course, topic, subtopic, difficulty, prerequisites, new_prereqs = (
                _render_taxonomy_controls(index, database, staged)
            )

            approved_question = build_enriched_question(
                base,
                course=course.strip(),
                topic=topic.strip(),
                subtopic=subtopic,
                difficulty=difficulty,
                prerequisites=[*prerequisites, *new_prereqs],
                source_metadata={"source_reference": result.source_reference},
            )
            # Reflejar la edición manual en el objeto que se usará al confirmar.
            approved_question = approved_question.model_copy(
                update={
                    "statement_raw": new_statement,
                    "question_type": qtype,
                    "choices": new_choices,
                }
            )
            approve = st.checkbox(
                "Aprobar y guardar esta pregunta",
                value=staged.status == STATUS_VALID,
                key=f"approve_{index}",
            )
            if approve:
                approved.append((approved_question, new_prereqs))

    st.divider()
    if st.button("Confirmar y persistir preguntas aprobadas", type="primary"):
        _confirm(approved, result, database, storage)


def _confirm(
    approved: list[tuple[EnrichedQuestion, list[str]]],
    result: StagingResult,
    database: Database,
    storage: ObjectStorage,
) -> None:
    if not approved:
        st.warning("No hay preguntas aprobadas para guardar.")
        return
    invalid = [
        q for q, _ in approved if not _SLUG_RE.match(q.course) or not _SLUG_RE.match(q.topic)
    ]
    if invalid:
        st.error("Curso y tema deben ser slugs válidos (minúsculas, dígitos y guion bajo).")
        return
    new_tags = sorted({slug for _, slugs in approved for slug in slugs})
    try:
        for question, _ in approved:
            database.upsert_course(question.course, question.course)
            database.upsert_topic(question.course, question.topic, question.subtopic)
        if new_tags:
            database.add_prerequisites(new_tags)
    except DatabaseError as exc:
        st.error(f"No se pudo actualizar la taxonomía: {exc}")
        return

    confirm_result = persist_approved_questions(
        [question for question, _ in approved],
        database,
        storage,
        session_id=result.session_id,
    )
    if confirm_result.saved:
        st.success(f"Persistidas {len(confirm_result.saved)} pregunta(s) en taygeta.db.")
    for error in confirm_result.errors:
        st.error(error)


def _sidebar(settings: Settings, database: Database, storage: ObjectStorage) -> None:
    with st.sidebar:
        st.header("Estado")
        try:
            status = database.status()
            st.write(
                {
                    "db": status["db_path"],
                    "integridad": status["integrity"],
                    "fts5": status["fts5_enabled"],
                    "preguntas": status["counts"]["questions"],
                }
            )
        except DatabaseError as exc:
            st.error(f"Base de datos no disponible: {exc}")
        st.caption(f"S3: {settings.S3_BUCKET} @ {settings.S3_ENDPOINT_URL}")
        if not settings.llm_configured:
            st.warning(
                "LLM remoto sin configurar: la validación semántica quedará en NEEDS_ATTENTION."
            )
        with st.expander("Taxonomía"):
            st.write("Cursos:", [c["slug"] for c in database.list_courses()])
            st.write("Prerrequisitos:", [p["slug"] for p in database.list_prerequisites()])


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------


def main() -> None:
    """Renderiza la aplicación Streamlit de revisión Human-in-the-Loop."""
    st.set_page_config(page_title="Taygeta — Revisión HITL", layout="wide")
    st.title("Taygeta — Ingesta y revisión Human-in-the-Loop")

    try:
        settings, database, storage, pipeline = _resources()
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudieron inicializar los servicios: {exc}")
        return
    st.session_state["_pipeline"] = pipeline
    _sidebar(settings, database, storage)

    st.subheader("1. Subir documento")
    uploaded = st.file_uploader(
        "PDF, PNG o JPG", type=UPLOAD_EXTENSIONS, accept_multiple_files=False
    )
    if uploaded is not None and st.button("Guardar en uploads"):
        path = save_upload(uploaded.name, uploaded.getvalue(), settings.UPLOADS_DIR)
        st.session_state["processed_path"] = str(path)
        st.success(f"Guardado en {path}")

    uploads = list_uploads(settings.UPLOADS_DIR)
    if not uploads:
        st.info("No hay documentos en el directorio de uploads.")
        return

    st.subheader("2. Procesar")
    selected_name = st.selectbox("Documento a procesar", [p.name for p in uploads])
    selected_path = next(path for path in uploads if path.name == selected_name)
    process_hint = st.text_area("Contexto para el análisis (opcional)", key="process_hint")
    if st.button("Procesar documento", type="primary"):
        with st.status("Ejecutando ingesta híbrida…", expanded=True) as status:
            st.write("Rasterizando y ejecutando el OCR local…")
            st.write("Estructurando con el LLM remoto y validando…")
            try:
                result = pipeline.ingest(selected_path, hints=process_hint or None)
            except Exception as exc:  # noqa: BLE001
                status.update(label="La ingesta falló", state="error")
                st.error(f"Error de ingesta: {exc}")
                return
            st.session_state["staging_result"] = result
            st.session_state["processed_path"] = str(selected_path)
            status.update(label=f"Ingesta completada: {result.status}", state="complete")

    result: StagingResult | None = st.session_state.get("staging_result")
    if result is None:
        return

    st.subheader("3. Revisión y confirmación")
    _render_review(result, settings, database, storage, st.session_state.get("processed_path"))


if __name__ == "__main__":
    main()
