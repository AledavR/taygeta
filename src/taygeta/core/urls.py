"""Utilidades para convertir rutas relativas de assets en URLs públicas.

Soporta la ejecución dual Local/Producción: los modelos almacenan las imágenes
como rutas relativas (p. ej. ``/buckets/exams/img.png``) y, justo antes de
enviar la información al frontend o al cliente MCP, se componen las URLs
completas concatenando ``S3_PUBLIC_BASE_URL``.
"""

from __future__ import annotations

from taygeta.core.schema import RELATIVE_ASSET_PREFIX, is_relative_asset_path
from taygeta.core.settings import Settings, get_settings

__all__ = [
    "build_public_url",
    "build_relative_path",
    "parse_relative_path",
    "relative_path_from_s3_uri",
    "split_asset_reference",
]

_S3_SCHEME = "s3://"


def build_relative_path(bucket: str, key: str) -> str:
    """Construye la ruta relativa canónica ``/buckets/{bucket}/{key}``.

    Raises:
        ValueError: si ``bucket`` o ``key`` quedan vacíos.
    """
    normalized_bucket = (bucket or "").strip("/")
    normalized_key = (key or "").strip("/")
    if not normalized_bucket or not normalized_key:
        raise ValueError("bucket y key no pueden quedar vacíos")
    return f"{RELATIVE_ASSET_PREFIX}{normalized_bucket}/{normalized_key}"


def parse_relative_path(path: str) -> tuple[str, str]:
    """Descompone ``/buckets/{bucket}/{key}`` en ``(bucket, key)``.

    Raises:
        ValueError: si ``path`` no es una ruta relativa de asset válida.
    """
    if not is_relative_asset_path(path):
        raise ValueError(f"ruta relativa de asset inválida: {path!r}")
    rest = path[len(RELATIVE_ASSET_PREFIX) :]
    bucket, _, key = rest.partition("/")
    return bucket, key


def relative_path_from_s3_uri(uri: str) -> str:
    """Convierte ``s3://bucket/key`` en ``/buckets/{bucket}/{key}``.

    Raises:
        ValueError: si ``uri`` no es una URI ``s3://`` válida.
    """
    if not uri.startswith(_S3_SCHEME):
        raise ValueError(f"URI de objeto inválida (se esperaba s3://): {uri!r}")
    rest = uri[len(_S3_SCHEME) :]
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"URI de objeto inválida: {uri!r}")
    return build_relative_path(bucket, key)


def split_asset_reference(
    reference: str, *, default_bucket: str | None = None
) -> tuple[str | None, str]:
    """Descompone una referencia de asset en ``(bucket, key)``.

    Acepta ``s3://bucket/key``, ``/buckets/{bucket}/{key}`` o una clave relativa
    al bucket por defecto. Para una clave relativa sin ``default_bucket``
    devuelve ``(None, key)`` y el llamador decide el bucket.

    Raises:
        ValueError: si la referencia está vacía o no se puede descomponer.
    """
    value = (reference or "").strip()
    if not value:
        raise ValueError("la referencia de asset no puede estar vacía")
    if value.startswith(_S3_SCHEME):
        rest = value[len(_S3_SCHEME) :]
        bucket, _, key = rest.partition("/")
        if not bucket or not key:
            raise ValueError(f"URI de objeto inválida: {reference!r}")
        return bucket, key
    if is_relative_asset_path(value):
        return parse_relative_path(value)
    if default_bucket:
        return default_bucket, value
    return None, value


def build_public_url(relative_path: str | None, settings: Settings | None = None) -> str:
    """Devuelve la URL pública completa de una ruta relativa de asset.

    Concatena ``S3_PUBLIC_BASE_URL`` (sin barra final) con la ruta relativa, de
    modo que ``/buckets/exams/img.png`` se convierte en, p. ej.,
    ``https://cdn.example.com/buckets/exams/img.png``. Si no hay base pública
    configurada, devuelve la ruta relativa tal cual (normalizada con ``/``).
    """
    if not relative_path:
        return ""
    resolved = settings or get_settings()
    base = (resolved.S3_PUBLIC_BASE_URL or "").rstrip("/")
    path = relative_path if relative_path.startswith("/") else "/" + relative_path
    return base + path if base else path
