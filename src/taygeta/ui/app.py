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

from taygeta.agents.enrichment import EnrichmentError, EnrichmentResult, QuestionEnricher
from taygeta.core.schema import (
    Choice,
    EnrichedQuestion,
    QuestionRaw,
    TranscriptionIntermediate,
    normalize_slug,
    slugify,
    utc_now_iso,
)
from taygeta.core.settings import Settings, get_settings
from taygeta.core.urls import build_public_url, build_relative_path, split_asset_reference
from taygeta.ingest.image_extractor import (
    is_supported_source,
    open_document,
    rasterize_page,
)
from taygeta.ingest.pipeline import IngestPipeline, StagedQuestion, StagingResult
from taygeta.ingest.validator import STATUS_NEEDS_ATTENTION, STATUS_VALID
from taygeta.server.db import Database, DatabaseError
from taygeta.server.object_storage import ObjectStorage, ObjectStorageError

logger = logging.getLogger(__name__)

__all__ = [
    "ConfirmResult",
    "build_enriched_question",
    "figure_bytes",
    "list_uploads",
    "main",
    "page_image_bytes",
    "persist_approved_questions",
    "public_s3_url",
    "sanitize_upload_name",
    "save_upload",
]

#: Extensiones aceptadas por el cargador de archivos.
UPLOAD_EXTENSIONS = ["pdf", "png", "jpg", "jpeg"]

_SLUG_RE = re.compile(r"^[a-z0-9_]+$")

#: Alias locales de los helpers compartidos de ``taygeta.core.schema``.
_slugify = slugify
_normalize_slug = normalize_slug


def _slug_split(text: str | None) -> list[str]:
    """Divide una cadena separada por comas en slugs normalizados (sin vacíos)."""
    if not text:
        return []
    return [slug for slug in (_slugify(part) for part in text.split(",")) if slug]


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

    Acepta ``/buckets/{bucket}/{key}``, ``s3://bucket/key`` o una clave de objeto.
    Devuelve ``None`` si la referencia está vacía o el objeto no se puede recuperar.
    """
    if storage is None or not asset_path:
        return None
    try:
        bucket, key = split_asset_reference(asset_path)
        if bucket:
            return storage.get_bytes(key, bucket=bucket)
        return storage.get_bytes(key)
    except (ObjectStorageError, ValueError) as exc:
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


def public_s3_url(url: str | None, settings: Settings) -> str:
    """Reescribe el endpoint interno de S3 por el público de la UI.

    Sustituye el prefijo ``S3_ENDPOINT_URL`` (p. ej. ``http://seaweedfs-s3:8333``)
    por ``S3_PUBLIC_ENDPOINT`` (p. ej. ``http://localhost:8333``) para poder abrir
    enlaces e imágenes desde el navegador del host. Si no hay endpoint público o
    la URL no usa el interno, se devuelve sin cambios.
    """
    if not url:
        return ""
    internal = (settings.S3_ENDPOINT_URL or "").rstrip("/")
    public = (settings.S3_PUBLIC_ENDPOINT or "").rstrip("/")
    if public and internal and url.startswith(internal):
        return public + url[len(internal) :]
    return url


def asset_browser_url(
    asset_path: str | None,
    settings: Settings,
    storage: ObjectStorage | None = None,
) -> str:
    """Devuelve una URL para abrir una figura desde el navegador del host.

    Orden de preferencia: ``S3_PUBLIC_BASE_URL`` (concatenada con la ruta
    relativa), ``ASSET_PUBLIC_BASE_URL`` (legacy), el proxy ``/assets/{key}`` de
    la aplicación y, en último caso, la URL reescrita del endpoint interno al
    público.
    """
    if not asset_path:
        return ""
    bucket, key = split_asset_reference(asset_path, default_bucket=settings.S3_BUCKET)
    if settings.S3_PUBLIC_BASE_URL:
        return build_public_url(build_relative_path(bucket, key), settings)
    if settings.ASSET_PUBLIC_BASE_URL:
        return f"{settings.ASSET_PUBLIC_BASE_URL.rstrip('/')}/{key}"
    proxy_base = (settings.DOWNLOAD_PUBLIC_URL or "").rstrip("/")
    if proxy_base:
        return f"{proxy_base}/assets/{key}"
    if storage is not None:
        try:
            return public_s3_url(storage.display_url(asset_path), settings)
        except ObjectStorageError:
            return ""
    return ""


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
    automática. ``course``, ``topic`` y ``subtopic`` se normalizan a slugs
    válidos (``^[a-z0-9_]+$``) para evitar ``ValidationError``; si ``course`` o
    ``topic`` quedan vacíos se usa ``"general"`` y el ``subtopic`` vacío se
    convierte en ``None``. Los prerrequisitos se normalizan igual y se descartan
    los que queden vacíos.
    """
    return EnrichedQuestion(
        id=base.id,
        statement_raw=base.statement_raw,
        question_type=base.question_type,
        choices=base.choices,
        figures=base.figures,
        notation_notes=base.notation_notes,
        course=_normalize_slug(course),
        topic=_normalize_slug(topic),
        subtopic=_slugify(subtopic) or None,
        difficulty=difficulty,
        prerequisites=[slug for slug in (_slugify(item) for item in prerequisites) if slug],
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
            bucket, key = split_asset_reference(figure.asset_path)
            if bucket:
                if not storage.exists(key, bucket=bucket):
                    missing.append(figure.asset_path)
            elif not storage.exists(key):
                missing.append(figure.asset_path)
        except (ObjectStorageError, ValueError):
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
def _resources() -> tuple[Settings, Database, ObjectStorage, IngestPipeline, QuestionEnricher]:
    settings = get_settings()
    database = Database(settings=settings)
    storage = ObjectStorage(settings)
    pipeline = IngestPipeline(settings, storage=storage, database=database)
    enricher = QuestionEnricher(settings)
    return settings, database, storage, pipeline, enricher


# ---------------------------------------------------------------------------
# Renderizado
# ---------------------------------------------------------------------------


def _render_reference_image(
    source_path: str | None,
    staged: StagedQuestion,
    storage: ObjectStorage | None,
    container: Any,
    settings: Settings,
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
        browser_url = asset_browser_url(figure.asset_path, settings, storage)
        if browser_url:
            container.caption(f"[Abrir {figure.id} en el navegador]({browser_url})")
    if shown or source_path is None:
        return
    pages = [0]
    if question is not None and question.figures:
        pages = sorted({figure.page for figure in question.figures})
    for page_index in pages:
        data = page_image_bytes(source_path, page_index)
        if data:
            container.image(data, caption=f"Página original {page_index}")


def _taxonomy_defaults(staged: StagedQuestion) -> dict[str, Any]:
    """Valores por defecto seguros (vía ``getattr``) para la taxonomía.

    El objeto en staging es habitualmente un ``QuestionRaw``, que **no** define
    ``course``/``topic``/``subtopic``/``difficulty``/``prerequisites`` (exclusivos
    de ``EnrichedQuestion``). Se accede con ``getattr`` para soportar ambos casos
    sin lanzar ``AttributeError``; si la pregunta ya está enriquecida, se
    reutilizan sus valores.
    """
    base = staged.question
    course = getattr(base, "course", "") or ""
    topic = getattr(base, "topic", "") or ""
    subtopic = getattr(base, "subtopic", None)
    difficulty = getattr(base, "difficulty", 3)
    prerequisites = getattr(base, "prerequisites", None) or []

    if not isinstance(subtopic, str) or not subtopic:
        subtopic = None
    if not isinstance(difficulty, int) or isinstance(difficulty, bool) or not 1 <= difficulty <= 5:
        difficulty = 3
    return {
        "course": course,
        "topic": topic,
        "subtopic": subtopic,
        "difficulty": difficulty,
        "prerequisites": list(prerequisites),
    }


def _render_taxonomy_controls(
    index: int,
    database: Database,
    staged: StagedQuestion,
    suggestion: EnrichmentResult | None = None,
) -> tuple[str, str, str | None, int, list[str], list[str]]:
    """Dibuja la taxonomía y devuelve los valores elegidos.

    Si se recibe una ``suggestion`` inferida por IA, se usa para **pre-rellenar**
    los controles (curso, tema, subtema, dificultad y prerrequisitos); en caso
    contrario se reutilizan los valores existentes de la pregunta.
    """
    defaults = _taxonomy_defaults(staged)
    suggested_course = suggestion.course if suggestion else ""
    suggested_topic = suggestion.topic if suggestion else ""
    suggested_subtopic = suggestion.subtopic if suggestion else None
    suggested_difficulty = suggestion.difficulty if suggestion else None
    suggested_prerequisites = list(suggestion.prerequisites) if suggestion else []

    course_default = suggested_course or defaults["course"]
    topic_default = suggested_topic or defaults["topic"]
    subtopic_default = suggested_subtopic or defaults["subtopic"] or ""
    difficulty_default = suggested_difficulty or defaults["difficulty"]

    # "Generación": al aplicar una nueva sugerencia se cambian las claves de los
    # widgets para que los nuevos valores por defecto se apliquen sin avisos de
    # ``session_state``.
    generation = st.session_state.get(f"taxonomy_gen_{index}", 0)

    def widget_key(name: str) -> str:
        return f"{name}_{index}_{generation}"

    courses = sorted({row["slug"] for row in database.list_courses()})
    course = st.text_input(
        "Curso (slug snake_case)",
        value=course_default,
        key=widget_key("course"),
        help=f"Cursos existentes: {', '.join(courses) or '(ninguno)'}",
    )
    topic = st.text_input(
        "Tema (slug snake_case)",
        value=topic_default,
        key=widget_key("topic"),
        help="Se crea en el catálogo al confirmar si no existe.",
    )
    subtopic_raw = st.text_input(
        "Subtema (slug snake_case, opcional)",
        value=subtopic_default,
        key=widget_key("subtopic"),
    )
    difficulty = st.slider(
        "Dificultad (1-5, complejidad matemática)",
        min_value=1,
        max_value=5,
        value=int(difficulty_default),
        key=widget_key("diff"),
    )

    prereq_default = list(dict.fromkeys([*suggested_prerequisites, *defaults["prerequisites"]]))
    catalog = sorted({row["slug"] for row in database.list_prerequisites()})
    options = sorted({*catalog, *prereq_default})
    prerequisites = st.multiselect(
        "Prerrequisitos",
        options,
        default=[slug for slug in prereq_default if slug in options],
        key=widget_key("prereq"),
    )
    new_prereqs_raw = st.text_input(
        "Nuevos prerrequisitos (separados por coma)", key=widget_key("new_prereq")
    )
    return (
        course,
        topic,
        _slugify(subtopic_raw) or None,
        difficulty,
        prerequisites,
        _slug_split(new_prereqs_raw),
    )


def _apply_enrichment(index: int, suggestion: EnrichmentResult) -> None:
    """Guarda la sugerencia y fuerza el re-prellenado de los controles."""
    st.session_state[f"enrichment_{index}"] = suggestion
    st.session_state[f"taxonomy_gen_{index}"] = st.session_state.get(f"taxonomy_gen_{index}", 0) + 1


def _enrich_all(result: StagingResult) -> None:
    """Infiere la taxonomía de todas las preguntas de la sesión de staging."""
    enricher = st.session_state.get("_enricher")
    if enricher is None:
        st.error("Enriquecedor no disponible (LLM no configurado).")
        return
    errors: list[str] = []
    for index, staged in enumerate(result.questions):
        if staged.question is None:
            continue
        try:
            suggestion = enricher.enrich(staged.question)
        except EnrichmentError as exc:
            errors.append(f"Pregunta {index + 1}: no se pudo inferir la taxonomía ({exc}).")
            continue
        _apply_enrichment(index, suggestion)
    if errors:
        st.session_state["_enrich_errors"] = errors
    st.rerun()


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

    for enrichment_error in st.session_state.pop("_enrich_errors", []):
        st.warning(enrichment_error)

    st.caption(
        "La taxonomía, la dificultad y los prerrequisitos pueden inferirse con IA y "
        "luego editarse manualmente antes de confirmar."
    )
    if st.button("Inferir taxonomía con IA (todas las preguntas)"):
        _enrich_all(result)

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
                _render_reference_image(source_path, staged, storage, st, settings)

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

            if st.button("Sugerir taxonomía con IA", key=f"enrich_{index}"):
                enricher = st.session_state.get("_enricher")
                if enricher is None:
                    st.error("Enriquecedor no disponible (LLM no configurado).")
                else:
                    try:
                        suggestion = enricher.enrich(base)
                    except EnrichmentError as exc:
                        st.error(f"No se pudo inferir la taxonomía: {exc}")
                    else:
                        _apply_enrichment(index, suggestion)
                        st.rerun()

            suggestion = st.session_state.get(f"enrichment_{index}")
            course, topic, subtopic, difficulty, prerequisites, new_prereqs = (
                _render_taxonomy_controls(index, database, staged, suggestion)
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
        public_endpoint = public_s3_url(settings.S3_ENDPOINT_URL, settings)
        st.caption(f"S3: {settings.S3_BUCKET} @ {public_endpoint}")
        if public_endpoint and public_endpoint != settings.S3_ENDPOINT_URL:
            st.caption(f"(endpoint interno: {settings.S3_ENDPOINT_URL})")
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


def _render_ingest_tab(
    settings: Settings,
    database: Database,
    storage: ObjectStorage,
    pipeline: IngestPipeline,
) -> None:
    """Pestaña de subida, procesamiento y revisión Human-in-the-Loop."""
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


def _render_question_bank(database: Database, settings: Settings) -> None:
    """Pestaña de exploración del banco de preguntas guardadas (SQLite + FTS5)."""
    st.subheader("Banco de preguntas")
    try:
        status = database.status()
    except DatabaseError as exc:
        st.error(f"Base de datos no disponible: {exc}")
        return
    st.metric("Preguntas registradas", status["counts"]["questions"])

    courses = ["(todos)", *sorted({c["slug"] for c in database.list_courses()})]
    course = st.selectbox("Filtrar por curso", courses, key="bank_course")
    course_filter = None if course == "(todos)" else course

    topics = ["(todos)"]
    if course_filter:
        topics += sorted({t["topic_slug"] for t in database.list_topics(course_filter)})
    topic = st.selectbox("Filtrar por tema", topics, key="bank_topic")
    topic_filter = None if topic == "(todos)" else topic

    query = st.text_input("Buscar en el enunciado (Markdown/TeX, FTS5)", key="bank_query").strip()
    limit = st.slider("Máximo de resultados", 5, 100, 20, step=5, key="bank_limit")
    try:
        questions = database.search_questions(
            query=query or None,
            course=course_filter,
            topic=topic_filter,
            limit=limit,
        )
    except DatabaseError as exc:
        st.error(f"Error en la búsqueda: {exc}")
        return

    st.caption(f"{len(questions)} resultado(s).")
    if not questions:
        st.info("No hay preguntas que cumplan los filtros.")
        return
    for question in questions:
        header = f"[{question.course}/{question.topic}] dificultad {question.difficulty}"
        with st.expander(header):
            st.markdown(question.statement_raw)
            st.json(
                {
                    "id": question.id,
                    "course": question.course,
                    "topic": question.topic,
                    "subtopic": question.subtopic,
                    "difficulty": question.difficulty,
                    "prerequisites": question.prerequisites,
                    "created_at": question.created_at,
                }
            )


def main() -> None:
    """Renderiza la aplicación Streamlit (ingesta/revisión + banco de preguntas)."""
    st.set_page_config(page_title="Taygeta — Revisión HITL", layout="wide")
    st.title("Taygeta — Ingesta y revisión Human-in-the-Loop")

    try:
        settings, database, storage, pipeline, enricher = _resources()
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudieron inicializar los servicios: {exc}")
        return
    st.session_state["_pipeline"] = pipeline
    st.session_state["_enricher"] = enricher
    _sidebar(settings, database, storage)

    tab_ingest, tab_bank = st.tabs(["Ingesta y revisión", "Banco de preguntas"])
    with tab_ingest:
        _render_ingest_tab(settings, database, storage, pipeline)
    with tab_bank:
        _render_question_bank(database, settings)


if __name__ == "__main__":
    main()
