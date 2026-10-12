"""Configuración tipada y validación de entorno para Taygeta.

Carga la configuración desde variables de entorno y, opcionalmente, desde un
archivo ``.env`` (nunca versionado). Mantiene separadas tres familias de
credenciales que no deben mezclarse:

* LLM remoto (``LLM_*``) para estructuración y validación lógica.
* OCR local OpenAI-compatible (``OCR_*``) para la extracción visual.
* Almacenamiento de objetos S3/SeaweedFS (``S3_*``).

Las claves nunca se registran en logs: usa :meth:`Settings.safe_summary`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings", "mask_secret"]


def mask_secret(value: str | None) -> str:
    """Devuelve una representación segura de un secreto para logs o resúmenes."""
    if not value:
        return "<empty>"
    if len(value) <= 4:
        return "****"
    return f"{value[:2]}****{value[-2:]}"


class Settings(BaseSettings):
    """Configuración de despliegue de Taygeta cargada desde el entorno."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # -- LLM remoto (estructuración y validación lógica) --------------------
    LLM_API_KEY: str = Field(
        default="",
        description="Clave API del proveedor remoto. Solo la credencial, no el endpoint.",
    )
    LLM_BASE_URL: str = Field(
        default="",
        description="URL base del endpoint remoto compatible con la API de OpenAI.",
    )
    LLM_MODEL: str = Field(
        default="",
        description="Identificador del modelo remoto elegido por el administrador.",
    )
    LLM_TIMEOUT_SECONDS: int = Field(default=120, ge=1, description="Timeout por solicitud LLM.")
    LLM_MAX_RETRIES: int = Field(default=2, ge=0, le=10, description="Reintentos acotados del LLM.")
    LLM_EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description=(
            "Modelo sentence-transformers (id de Hugging Face o ruta local) para "
            "generar embeddings de deduplicación de forma local, sin API externa. "
            "Se recomienda un modelo multilingüe para texto en español."
        ),
    )

    # -- Deduplicación semántica --------------------------------------------
    DUPLICATE_SIMILARITY_THRESHOLD: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description=(
            "Umbral de similitud coseno para marcar una pregunta como candidata a "
            "duplicado durante la ingesta."
        ),
    )

    # -- OCR local (OpenAI-compatible) --------------------------------------
    OCR_API_KEY: str = Field(
        default="local-ocr",
        description="Credencial requerida por el cliente para el servicio OCR local.",
    )
    OCR_BASE_URL: str = Field(
        default="http://ocr:8080/v1",
        description="Endpoint OpenAI-compatible del servidor OCR local.",
    )
    OCR_MODEL: str = Field(
        default="ggml-org/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M",
        description="Identificador de modelo que acepta el servidor OCR local.",
    )
    OCR_MODEL_PATH: str = Field(
        default="",
        description="Ruta del modelo/pesos dentro del volumen persistente del OCR.",
    )
    OCR_TIMEOUT_SECONDS: int = Field(
        default=180, ge=1, description="Timeout para una solicitud de OCR."
    )
    OCR_CONTEXT_SIZE: int = Field(
        default=8192, ge=512, description="Longitud de contexto del modelo OCR local."
    )

    # -- Entorno de ejecución y base de datos -------------------------------
    APP_ENV: Literal["local", "production"] = Field(
        default="local",
        description="Entorno de ejecución: 'local' o 'production'.",
    )
    DATABASE_URL: str = Field(
        default="",
        description=(
            "URL de la base de datos (p. ej. sqlite:////app/data/taygeta.db). "
            "Si está vacía, se usa DB_PATH."
        ),
    )

    # -- SQLite y directorios runtime ---------------------------------------
    DB_PATH: str = Field(default="/app/data/taygeta.db", description="Ruta de la base SQLite.")
    DOWNLOADS_DIR: str = Field(
        default="/app/downloads", description="Directorio de entregables temporales."
    )
    UPLOADS_DIR: str = Field(
        default="/app/uploads", description="Directorio de documentos recibidos."
    )
    DOWNLOADS_TTL_SECONDS: int = Field(
        default=3600,
        ge=0,
        description="TTL de limpieza de /app/downloads (60 minutos por defecto).",
    )

    # -- Almacenamiento de objetos S3/SeaweedFS -----------------------------
    S3_ENDPOINT_URL: str = Field(
        default="http://seaweedfs-s3:8333", description="Endpoint interno S3."
    )
    S3_PUBLIC_ENDPOINT: str = Field(
        default="http://localhost:8333",
        description=(
            "Endpoint S3 público/externo para la UI (reescribe el host interno en "
            "enlaces e imágenes para que sean accesibles desde el navegador)."
        ),
    )
    S3_PUBLIC_BASE_URL: str = Field(
        default="http://localhost:8333",
        description=(
            "Base pública para componer URLs de assets a partir de rutas relativas "
            "(p. ej. https://assets.example.com). Se concatena con la ruta relativa "
            "/buckets/{bucket}/{key} al enviar la información al frontend o al MCP."
        ),
    )
    S3_BUCKET: str = Field(default="taygeta-assets", description="Bucket de figuras.")
    S3_ACCESS_KEY_ID: str = Field(default="change-me", description="Credencial de acceso S3.")
    S3_SECRET_ACCESS_KEY: str = Field(default="change-me", description="Clave secreta S3.")
    S3_REGION: str = Field(default="us-east-1", description="Región compatible con el cliente S3.")
    ASSET_PUBLIC_BASE_URL: str = Field(
        default="", description="Base pública opcional o endpoint de proxy para la UI."
    )
    ASSET_URL_TTL_SECONDS: int = Field(
        default=300, ge=1, description="Duración de las URLs prefirmadas de assets."
    )

    # -- Direcciones públicas de los servicios ------------------------------
    MCP_HOST: str = Field(default="0.0.0.0", description="Host de escucha del servidor MCP.")
    MCP_PORT: int = Field(default=8000, ge=1, le=65535, description="Puerto del servidor MCP.")
    STREAMLIT_PUBLIC_URL: str = Field(
        default="http://localhost:8501", description="URL pública de la interfaz Streamlit."
    )
    DOWNLOAD_PUBLIC_URL: str = Field(
        default="http://localhost:8000", description="URL pública del servidor de descargas."
    )

    # -- Validadores --------------------------------------------------------
    @field_validator("APP_ENV", mode="before")
    @classmethod
    def _normalize_app_env(cls, value: object) -> str:
        normalized = str(value).strip().lower() if value is not None else ""
        if normalized not in ("local", "production"):
            raise ValueError("APP_ENV debe ser 'local' o 'production'")
        return normalized

    @field_validator("DB_PATH", "DOWNLOADS_DIR", "UPLOADS_DIR")
    @classmethod
    def _validate_runtime_paths(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("la ruta no puede estar vacía")
        return value

    @field_validator(
        "LLM_BASE_URL",
        "OCR_BASE_URL",
        "S3_ENDPOINT_URL",
        "S3_PUBLIC_ENDPOINT",
        "S3_PUBLIC_BASE_URL",
        "ASSET_PUBLIC_BASE_URL",
    )
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        if value and value != "/":
            return value.rstrip("/")
        return value

    # -- Propiedades y utilidades -------------------------------------------
    @property
    def llm_configured(self) -> bool:
        """Indica si el LLM remoto tiene credencial, endpoint y modelo completos."""
        return bool(self.LLM_API_KEY and self.LLM_BASE_URL and self.LLM_MODEL)

    @property
    def ocr_configured(self) -> bool:
        """Indica si el OCR local tiene endpoint y modelo definidos."""
        return bool(self.OCR_BASE_URL and self.OCR_MODEL)

    @property
    def s3_configured(self) -> bool:
        """Indica si el almacenamiento S3 tiene bucket y credenciales definidos."""
        return bool(
            self.S3_ENDPOINT_URL
            and self.S3_BUCKET
            and self.S3_ACCESS_KEY_ID
            and self.S3_SECRET_ACCESS_KEY
        )

    @property
    def is_production(self) -> bool:
        """Indica si la aplicación se ejecuta en el entorno de producción."""
        return self.APP_ENV == "production"

    @property
    def database_path(self) -> str:
        """Ruta efectiva de la base de datos.

        Si ``DATABASE_URL`` está configurada como URL ``sqlite://``, extrae la
        ruta de la misma; en caso contrario usa ``DB_PATH``. Las URLs no SQLite
        (p. ej. ``postgresql://``) no están soportadas por la capa SQLite actual
        y provocan un ``ValueError`` explícito.
        """
        url = (self.DATABASE_URL or "").strip()
        if not url:
            return self.DB_PATH
        if url == "sqlite://":
            return ":memory:"
        if url.startswith("sqlite:///"):
            # sqlite:///rel.db  -> "rel.db"        (ruta relativa)
            # sqlite:////abs.db -> "/abs.db"       (ruta absoluta)
            # sqlite:///:memory: -> ":memory:"
            return url[len("sqlite://") :][1:]
        raise ValueError(
            f"DATABASE_URL no soportada: {url!r}; la capa actual solo admite sqlite://"
        )

    def require_llm(self) -> None:
        """Valida que la configuración del LLM remoto esté completa.

        Raises:
            RuntimeError: si falta alguna de las variables obligatorias.
        """
        missing = [
            name for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL") if not getattr(self, name)
        ]
        if missing:
            raise RuntimeError(
                "configuración del LLM remoto incompleta; define: " + ", ".join(missing)
            )

    def ensure_runtime_directories(self) -> list[Path]:
        """Crea los directorios runtime necesarios y devuelve sus rutas.

        No crea el directorio de la base de datos si su ruta es relativa al
        repositorio; en ese caso devuelve igualmente las rutas implicadas.
        """
        directories = [
            Path(self.DB_PATH).expanduser().parent,
            Path(self.DOWNLOADS_DIR).expanduser(),
            Path(self.UPLOADS_DIR).expanduser(),
        ]
        created: list[Path] = []
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            created.append(directory)
        return created

    def safe_summary(self) -> dict[str, object]:
        """Resumen apto para logs, con todas las credenciales enmascaradas."""
        return {
            "app_env": self.APP_ENV,
            "database_url": mask_secret(self.DATABASE_URL),
            "llm_configured": self.llm_configured,
            "llm_base_url": self.LLM_BASE_URL,
            "llm_model": self.LLM_MODEL,
            "llm_timeout_seconds": self.LLM_TIMEOUT_SECONDS,
            "llm_max_retries": self.LLM_MAX_RETRIES,
            "llm_embedding_model": self.LLM_EMBEDDING_MODEL,
            "duplicate_similarity_threshold": self.DUPLICATE_SIMILARITY_THRESHOLD,
            "ocr_base_url": self.OCR_BASE_URL,
            "ocr_model": self.OCR_MODEL,
            "ocr_timeout_seconds": self.OCR_TIMEOUT_SECONDS,
            "ocr_context_size": self.OCR_CONTEXT_SIZE,
            "db_path": self.DB_PATH,
            "downloads_dir": self.DOWNLOADS_DIR,
            "uploads_dir": self.UPLOADS_DIR,
            "downloads_ttl_seconds": self.DOWNLOADS_TTL_SECONDS,
            "s3_endpoint_url": self.S3_ENDPOINT_URL,
            "s3_public_endpoint": self.S3_PUBLIC_ENDPOINT,
            "s3_public_base_url": self.S3_PUBLIC_BASE_URL,
            "s3_bucket": self.S3_BUCKET,
            "s3_region": self.S3_REGION,
            "s3_access_key_id": mask_secret(self.S3_ACCESS_KEY_ID),
            "s3_secret_access_key": mask_secret(self.S3_SECRET_ACCESS_KEY),
            "llm_api_key": mask_secret(self.LLM_API_KEY),
            "ocr_api_key": mask_secret(self.OCR_API_KEY),
            "mcp_host": self.MCP_HOST,
            "mcp_port": self.MCP_PORT,
            "streamlit_public_url": self.STREAMLIT_PUBLIC_URL,
            "download_public_url": self.DOWNLOAD_PUBLIC_URL,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devuelve una instancia única y en caché de :class:`Settings`."""
    return Settings()
