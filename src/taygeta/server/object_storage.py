"""Almacenamiento de objetos S3 (SeaweedFS) para figuras y assets de Taygeta.

Secci\u00f3n 7 de la especificaci\u00f3n. Centraliza el ciclo de vida de los objetos:

* crear el bucket si corresponde, subir, descargar, consultar metadatos,
  generar URLs prefirmadas y borrar;
* almacenar en la base de datos una clave estable o una URI ``s3://bucket/key``,
  nunca una URL prefirmada temporal ni una ruta local absoluta;
* validar identificadores para evitar traversal en el proxy HTTP de figuras;
* permitir que la UI muestre figuras sin conocer rutas internas.

El cliente se construye con boto3 y direccionamiento *path-style*, adecuado para
la puerta de enlace S3 de SeaweedFS.
"""

from __future__ import annotations

import mimetypes
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO
from urllib.parse import quote

import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError

from taygeta.core.settings import Settings, get_settings

if TYPE_CHECKING:
    pass  # type: ignore[import-not-found]

__all__ = [
    "ObjectStorage",
    "ObjectStorageError",
    "build_s3_uri",
    "get_object_storage",
    "guessed_content_type",
    "parse_s3_uri",
    "sanitize_asset_id",
    "validate_object_key",
]

_S3_SCHEME = "s3://"
_ASSET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


class ObjectStorageError(RuntimeError):
    """Error controlado del almacenamiento de objetos S3."""


# ---------------------------------------------------------------------------
# Validaci\u00f3n de claves e identificadores
# ---------------------------------------------------------------------------


def validate_object_key(key: str) -> str:
    """Valida una clave de objeto relativa al bucket.

    Rechaza rutas absolutas, barras invertidas, caracteres de control y
    segmentos ``.`` / ``..`` para impedir traversal. Devuelve la clave validada.

    Raises:
        ObjectStorageError: si la clave no es segura.
    """
    if not isinstance(key, str) or not key.strip():
        raise ObjectStorageError("la clave de objeto no puede estar vacía")
    if key != key.strip():
        raise ObjectStorageError("la clave de objeto no puede tener espacios al inicio o al final")
    if key.startswith("/") or key.startswith("\\"):
        raise ObjectStorageError("la clave de objeto no puede ser una ruta absoluta")
    if "\\" in key:
        raise ObjectStorageError("la clave de objeto no puede contener barras invertidas")
    if _CONTROL_CHARS_RE.search(key):
        raise ObjectStorageError("la clave de objeto no puede contener caracteres de control")
    segments = key.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        raise ObjectStorageError(
            "la clave de objeto no puede contener segmentos vacíos, '.' o '..'"
        )
    return key


def sanitize_asset_id(asset_id: str) -> str:
    """Valida un identificador de asset de un solo segmento (para el proxy HTTP)."""
    if not isinstance(asset_id, str):
        raise ObjectStorageError("el identificador de asset debe ser texto")
    candidate = asset_id.strip()
    if not _ASSET_ID_RE.match(candidate):
        raise ObjectStorageError(
            "identificador de asset inválido; solo se permiten letras, dígitos, '.', '_' y '-'"
        )
    return candidate


def build_s3_uri(bucket: str, key: str) -> str:
    """Construye una URI estable ``s3://bucket/key``."""
    if not bucket:
        raise ObjectStorageError("el bucket no puede estar vacío")
    return f"{_S3_SCHEME}{bucket}/{validate_object_key(key)}"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Descompone ``s3://bucket/key`` en ``(bucket, key)`` validando la clave."""
    if not uri.startswith(_S3_SCHEME):
        raise ObjectStorageError(f"URI de objeto inválida (se esperaba s3://): {uri!r}")
    remainder = uri[len(_S3_SCHEME) :]
    bucket, _, key = remainder.partition("/")
    if not bucket or not key:
        raise ObjectStorageError(f"URI de objeto inválida: {uri!r}")
    return bucket, validate_object_key(key)


def guessed_content_type(key: str) -> str:
    """Infiere el tipo MIME a partir de la extensión de la clave."""
    content_type, _ = mimetypes.guess_type(key)
    return content_type or "application/octet-stream"


# ---------------------------------------------------------------------------
# Cliente de almacenamiento
# ---------------------------------------------------------------------------


class ObjectStorage:
    """Cliente S3 de alto nivel para el bucket de figuras de Taygeta."""

    def __init__(self, settings: Settings | None = None, *, client: Any | None = None) -> None:
        """Crea el cliente.

        Args:
            settings: Configuración a usar; por defecto la global.
            client: Cliente boto3 ya construido (inyección para pruebas
                unitarias). Si se omite, se construye uno real de forma perezosa.
        """
        self.settings = settings or get_settings()
        self.bucket = self.settings.S3_BUCKET
        self._client = client

    # -- Cliente ------------------------------------------------------------
    @property
    def client(self) -> Any:
        """Cliente boto3 S3, construido de forma perezosa la primera vez."""
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> Any:
        config = Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            connect_timeout=10,
            read_timeout=60,
            retries={"max_attempts": 3, "mode": "standard"},
        )
        return boto3.session.Session().client(
            "s3",
            endpoint_url=self.settings.S3_ENDPOINT_URL,
            aws_access_key_id=self.settings.S3_ACCESS_KEY_ID,
            aws_secret_access_key=self.settings.S3_SECRET_ACCESS_KEY,
            region_name=self.settings.S3_REGION,
            config=config,
        )

    # -- Bucket -------------------------------------------------------------
    def ensure_bucket(self, bucket: str | None = None) -> bool:
        """Crea el bucket si no existe. Devuelve ``True`` si tuvo que crearlo."""
        target = bucket or self.bucket
        if not target:
            raise ObjectStorageError("no se configuró ningún bucket")
        try:
            self.client.head_bucket(Bucket=target)
            return False
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code not in ("404", "NoSuchBucket", "NotFound") and status != 404:
                raise self._wrap_error("head_bucket", target, exc) from exc

        params: dict[str, Any] = {"Bucket": target}
        if self.settings.S3_REGION and self.settings.S3_REGION != "us-east-1":
            params["CreateBucketConfiguration"] = {"LocationConstraint": self.settings.S3_REGION}
        try:
            self.client.create_bucket(**params)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                raise self._wrap_error("create_bucket", target, exc) from exc
        return True

    # -- Escritura ----------------------------------------------------------
    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
        bucket: str | None = None,
    ) -> str:
        """Sube ``data`` a ``key`` y devuelve la URI estable ``s3://bucket/key``."""
        target = bucket or self.bucket
        validate_object_key(key)
        try:
            self.client.put_object(
                Bucket=target,
                Key=key,
                Body=data,
                ContentType=content_type or guessed_content_type(key),
                Metadata=metadata or {},
            )
        except (ClientError, BotoCoreError) as exc:
            raise self._wrap_error("put_object", key, exc) from exc
        return build_s3_uri(target, key)

    def put_file(
        self,
        key: str,
        path: str | Path,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
        bucket: str | None = None,
    ) -> str:
        """Sube el archivo ``path`` a ``key`` y devuelve la URI estable."""
        source = Path(path)
        if not source.is_file():
            raise ObjectStorageError(f"el archivo de origen no existe: {source}")
        return self.put_bytes(
            key,
            source.read_bytes(),
            content_type=content_type,
            metadata=metadata,
            bucket=bucket,
        )

    def put_stream(
        self,
        key: str,
        stream: BinaryIO,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
        bucket: str | None = None,
    ) -> str:
        """Sube un flujo binario a ``key`` y devuelve la URI estable."""
        target = bucket or self.bucket
        validate_object_key(key)
        try:
            self.client.upload_fileobj(
                stream,
                target,
                key,
                ExtraArgs={
                    "ContentType": content_type or guessed_content_type(key),
                    "Metadata": metadata or {},
                },
            )
        except (ClientError, BotoCoreError) as exc:
            raise self._wrap_error("upload_fileobj", key, exc) from exc
        return build_s3_uri(target, key)

    # -- Lectura ------------------------------------------------------------
    def get_bytes(self, key: str, *, bucket: str | None = None) -> bytes:
        """Descarga y devuelve los bytes del objeto ``key``."""
        target = bucket or self.bucket
        validate_object_key(key)
        try:
            response = self.client.get_object(Bucket=target, Key=key)
            return response["Body"].read()
        except (ClientError, BotoCoreError) as exc:
            raise self._wrap_error("get_object", key, exc) from exc

    def download_to_file(
        self,
        key: str,
        destination: str | Path,
        *,
        bucket: str | None = None,
    ) -> Path:
        """Descarga el objeto ``key`` en ``destination`` y devuelve la ruta."""
        target = bucket or self.bucket
        validate_object_key(key)
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.client.download_file(target, key, str(destination_path))
        except (ClientError, BotoCoreError) as exc:
            raise self._wrap_error("download_file", key, exc) from exc
        return destination_path

    def exists(self, key: str, *, bucket: str | None = None) -> bool:
        """Indica si el objeto existe."""
        target = bucket or self.bucket
        validate_object_key(key)
        try:
            self.client.head_object(Bucket=target, Key=key)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in ("404", "NoSuchKey", "NotFound") or status == 404:
                return False
            raise self._wrap_error("head_object", key, exc) from exc

    def head(self, key: str, *, bucket: str | None = None) -> dict[str, Any]:
        """Devuelve los metadatos del objeto (tamaño, tipo, ETag, etc.)."""
        target = bucket or self.bucket
        validate_object_key(key)
        try:
            response = self.client.head_object(Bucket=target, Key=key)
        except (ClientError, BotoCoreError) as exc:
            raise self._wrap_error("head_object", key, exc) from exc
        return {
            "bucket": target,
            "key": key,
            "size": response.get("ContentLength"),
            "content_type": response.get("ContentType"),
            "etag": response.get("ETag"),
            "last_modified": response.get("LastModified"),
            "metadata": response.get("Metadata", {}),
        }

    # -- Borrado ------------------------------------------------------------
    def delete(self, key: str, *, bucket: str | None = None) -> None:
        """Borra el objeto ``key`` de forma idempotente (no falla si no existe)."""
        target = bucket or self.bucket
        validate_object_key(key)
        try:
            self.client.delete_object(Bucket=target, Key=key)
        except (ClientError, BotoCoreError) as exc:
            raise self._wrap_error("delete_object", key, exc) from exc

    def delete_uri(self, uri: str) -> None:
        """Borra el objeto referenciado por una URI ``s3://bucket/key``."""
        bucket, key = parse_s3_uri(uri)
        self.delete(key, bucket=bucket)

    # -- URLs ---------------------------------------------------------------
    def presigned_url(
        self,
        key: str,
        *,
        expires_in: int | None = None,
        bucket: str | None = None,
    ) -> str:
        """Genera una URL prefirmada de vida limitada para leer el objeto.

        No se debe persistir en la base de datos; se genera bajo demanda. El
        host de la URL no se reescribe tras firmarla, porque eso invalidaría la
        firma.
        """
        target = bucket or self.bucket
        validate_object_key(key)
        ttl = expires_in or self.settings.ASSET_URL_TTL_SECONDS
        try:
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": target, "Key": key},
                ExpiresIn=ttl,
            )
        except (ClientError, BotoCoreError) as exc:
            raise self._wrap_error("generate_presigned_url", key, exc) from exc

    def display_url(self, reference: str) -> str:
        """Devuelve una URL apta para la UI a partir de una clave o URI estable.

        Si ``ASSET_PUBLIC_BASE_URL`` está configurada, compone una URL pública
        sobre la clave; en caso contrario, genera una URL prefirmada temporal.
        La UI no necesita conocer rutas de almacenamiento internas.
        """
        bucket, key = self._split_reference(reference)
        public_base = self.settings.ASSET_PUBLIC_BASE_URL
        if public_base:
            return f"{public_base.rstrip('/')}/{quote(key)}"
        return self.presigned_url(key, bucket=bucket)

    # -- Assets -------------------------------------------------------------
    def generate_asset_key(
        self,
        prefix: str = "figures",
        *,
        asset_id: str | None = None,
        extension: str = ".png",
    ) -> str:
        """Genera una clave de objeto estable y segura para un asset."""
        normalized_prefix = prefix.strip("/")
        if normalized_prefix:
            validate_object_key(normalized_prefix)
        suffix = extension if extension.startswith(".") else f".{extension}"
        suffix = re.sub(r"[^A-Za-z0-9.]", "", suffix) or ".png"
        name = sanitize_asset_id(asset_id) if asset_id else uuid.uuid4().hex
        key = f"{normalized_prefix}/{name}{suffix}" if normalized_prefix else f"{name}{suffix}"
        return validate_object_key(key)

    def store_asset_bytes(
        self,
        data: bytes,
        *,
        prefix: str = "figures",
        asset_id: str | None = None,
        extension: str = ".png",
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Sube bytes como asset con clave estable y devuelve la URI ``s3://...``."""
        key = self.generate_asset_key(prefix, asset_id=asset_id, extension=extension)
        return self.put_bytes(key, data, content_type=content_type, metadata=metadata)

    def store_asset_file(
        self,
        path: str | Path,
        *,
        prefix: str = "figures",
        asset_id: str | None = None,
        extension: str = ".png",
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Sube un archivo como asset con clave estable y devuelve la URI."""
        source = Path(path)
        return self.store_asset_bytes(
            source.read_bytes(),
            prefix=prefix,
            asset_id=asset_id,
            extension=extension or source.suffix or ".png",
            content_type=content_type,
            metadata=metadata,
        )

    def resolve(self, reference: str) -> tuple[str, str]:
        """Resuelve una referencia estable a ``(bucket, key)`` validado.

        Acepta tanto ``s3://bucket/key`` como una clave relativa al bucket
        configurado. Rechaza traversal y rutas absolutas, por lo que es apto
        para el endpoint proxy ``/assets/{asset_id}``.
        """
        return self._split_reference(reference)

    def _split_reference(self, reference: str) -> tuple[str, str]:
        value = (reference or "").strip()
        if not value:
            raise ObjectStorageError("la referencia de asset no puede estar vacía")
        if value.startswith(_S3_SCHEME):
            return parse_s3_uri(value)
        return self.bucket, validate_object_key(value)

    @staticmethod
    def _wrap_error(operation: str, key: str, exc: Exception) -> ObjectStorageError:
        return ObjectStorageError(f"fallo en {operation} sobre el objeto {key!r}: {exc}")


def get_object_storage() -> ObjectStorage:
    """Devuelve un cliente de almacenamiento de objetos con la configuración global."""
    return ObjectStorage(get_settings())
