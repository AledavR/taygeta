"""Helpers compartidos por la UI HITL de Taygeta (independientes del framework).

Concentra la lógica reutilizable de la UI:

* subida/guardado de documentos y listado del directorio de subidas;
* rasterización de páginas y recuperación de figuras desde SeaweedFS;
* composición de URLs públicas (endpoints interno/público de S3);
* construcción de :class:`EnrichedQuestion` a partir de los campos revisados
  y persistencia de preguntas aprobadas (``persist_approved_questions``).

Este módulo no importa NiceGUI, por lo que es unit-testable y reutilizable
desde cualquier capa.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from taygeta.agents.embeddings import Embedder, question_embedding_text
from taygeta.core.schema import (
    EnrichedQuestion,
    FigureReference,
    QuestionRaw,
    get_public_url,
    normalize_slug,
    slugify,
    utc_now_iso,
)
from taygeta.core.settings import get_settings
from taygeta.core.urls import build_relative_path, split_asset_reference
from taygeta.ingest.image_extractor import (
    PageRaster,
    is_supported_source,
    open_document,
    rasterize_page,
)
from taygeta.server.db import Database, DatabaseError
from taygeta.server.object_storage import ObjectStorage, ObjectStorageError

logger = logging.getLogger(__name__)

__all__ = [
    "ConfirmResult",
    "build_enriched_question",
    "cache_busted_public_url",
    "delete_asset",
    "document_page_count",
    "figure_bytes",
    "list_uploads",
    "page_image_bytes",
    "page_image_raster",
    "persist_approved_questions",
    "public_asset_url",
    "public_s3_url",
    "sanitize_upload_name",
    "save_upload",
    "slug_split",
]

#: Alias locales de los helpers compartidos de ``taygeta.core.schema``.
_slugify = slugify
_normalize_slug = normalize_slug


def slug_split(text: str | None) -> list[str]:
    """Divide una cadena separada por comas en slugs normalizados (sin vacíos)."""
    if not text:
        return []
    return [slug for slug in (_slugify(part) for part in text.split(",")) if slug]


# ---------------------------------------------------------------------------
# Utilidades de archivos y figuras (testables sin framework)
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


def page_image_raster(
    source_path: str | Path, page_index: int, *, dpi: int = 150
) -> PageRaster | None:
    """Devuelve la ``PageRaster`` de una página para el cropper (bytes + tamaño).

    ``PageRaster`` expone ``png_bytes`` y ``width_px``/``height_px``, que se
    necesitan tanto para alimentar el recorte manual como para convertir el
    recuadro de píxeles a la BBox normalizada.
    """
    try:
        with open_document(source_path) as document:
            if page_index < 0 or page_index >= document.page_count:
                return None
            page = document.load_page(page_index)
            is_pdf = bool(getattr(document, "is_pdf", False))
            return rasterize_page(page, dpi=dpi, is_pdf=is_pdf)
    except Exception as exc:  # noqa: BLE001 - la vista previa nunca debe romper la UI
        logger.warning("no se pudo rasterizar la página %s: %s", page_index, exc)
        return None


def document_page_count(source_path: str | Path | None) -> int | None:
    """Devuelve el número de páginas del documento origen, o ``None`` si es inaccesible.

    Se usa para ofrecer el selector de página del recorte manual de figuras.
    """
    if source_path is None:
        return None
    try:
        with open_document(source_path) as document:
            return document.page_count
    except Exception as exc:  # noqa: BLE001 - la UI nunca debe romper por el origen
        logger.warning("no se pudo contar las páginas de %s: %s", source_path, exc)
        return None


def delete_asset(storage: ObjectStorage | None, asset_path: str) -> None:
    """Borra un asset por su referencia estable (best-effort, nunca bloquea)."""
    if storage is None:
        return
    try:
        bucket, key = split_asset_reference(asset_path, default_bucket=storage.bucket)
        storage.delete(key, bucket=bucket or None)
    except (ObjectStorageError, ValueError) as exc:
        logger.warning("no se pudo purgar el asset huérfano %s: %s", asset_path, exc)


def public_s3_url(url: str | None, settings: Any) -> str:
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


def public_asset_url(asset_path: str | None, settings: Any) -> str:
    """Compone la URL pública de una figura con ``get_public_url`` de schema.

    Convierte cualquier referencia de asset (``/buckets/{bucket}/{key}``,
    ``s3://bucket/key`` o clave relativa) a su ruta relativa canónica y la
    concatena con ``S3_PUBLIC_BASE_URL``. Así la UI renderiza imágenes con la URL
    pública del host, nunca con URLs internas de Docker (p. ej.
    ``http://seaweedfs-s3:8333``).
    """
    if not asset_path:
        return ""
    try:
        bucket, key = split_asset_reference(asset_path, default_bucket=settings.S3_BUCKET)
        relative = build_relative_path(bucket, key)
    except ValueError:
        return ""
    return get_public_url(relative)


def cache_busted_public_url(asset_path: str | None, settings: Any, seed: Any = "") -> str:
    """URL pública de una figura con parámetro de versión anti-caché.

    La vista previa lee los bytes directamente de SeaweedFS (sin caché de
    navegador); el enlace externo añade ``?v=`` derivado del asset y de la BBox
    para que el navegador no sirva una versión antigua tras un recorte.
    """
    url = public_asset_url(asset_path, settings)
    if not url:
        return ""
    token = hashlib.sha256(f"{asset_path}|{seed}".encode()).hexdigest()[:8]
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}v={token}"


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
    figures: Sequence[FigureReference] | None = None,
) -> EnrichedQuestion:
    """Compone una :class:`EnrichedQuestion` a partir de los campos revisados.

    Los valores editados manualmente prevalecen sobre la transcripción
    automática. ``course``, ``topic`` y ``subtopic`` se normalizan a slugs
    válidos (``^[a-z0-9_]+$``) para evitar ``ValidationError``; si ``course`` o
    ``topic`` quedan vacíos se usa ``"general"`` y el ``subtopic`` vacío se
    convierte en ``None``. Los prerrequisitos se normalizan igual y se descartan
    los que queden vacíos. ``figures`` permite sustituir las figuras de
    ``base.figures`` con la selección editada en la revisión (borrados y
    recortes); si se omite, se conservan las originales.
    """
    return EnrichedQuestion(
        id=base.id,
        statement_raw=base.statement_raw,
        question_type=base.question_type,
        choices=base.choices,
        figures=figures if figures is not None else base.figures,
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
    embedder: Embedder | None = None,
) -> ConfirmResult:
    """Persiste las preguntas aprobadas tras verificar sus figuras en el bucket.

    Usa ``upsert_question`` para que la confirmación sea idempotente. Solo marca
    la sesión de staging como ``CONFIRMED`` cuando ha guardado alguna pregunta.
    Genera (best-effort) el embedding de cada pregunta para la deduplicación
    semántica posterior; si el proveedor no está disponible, se omite sin
    bloquear la confirmación.
    """
    if embedder is None:
        embedder = Embedder(get_settings())
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
            database.upsert_question(question, embedding=_embed_or_none(embedder, question))
            result.saved.append(question.id)
        except Exception as exc:  # noqa: BLE001 - se informa al usuario, no se oculta
            result.errors.append(f"{question.id}: {exc}")

    if session_id and result.saved:
        try:
            database.update_staging_session(session_id, status="CONFIRMED")
        except DatabaseError as exc:
            result.errors.append(f"no se pudo marcar la sesión como confirmada: {exc}")
    return result


def _embed_or_none(embedder: Embedder, question: EnrichedQuestion) -> list[float] | None:
    """Genera el embedding de una pregunta sin elevar ante fallos del proveedor."""
    try:
        return embedder.embed(question_embedding_text(question))
    except Exception:  # noqa: BLE001 - la deduplicación semántica es opcional
        logger.warning("no se pudo generar el embedding de la pregunta %s", question.id)
        return None
