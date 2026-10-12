"""Pase 1: extracción visual local mediante OCR/Vision-LLM (sección 4.2).

El cliente se comunica con un servicio local que expone una API compatible con
OpenAI, configurado exclusivamente con las variables ``OCR_BASE_URL``,
``OCR_API_KEY`` y ``OCR_MODEL``. Su responsabilidad es *transcribir fielmente*
el contenido visible a Markdown bruto con matemáticas TeX (``$...$`` y
``$$...$$``), marcando los segmentos dudosos con ``[ILEGIBLE]``.

No clasifica curricularmente, no asigna dificultad, no infiere respuestas
correctas, no resuelve ejercicios ni produce el objeto final ``QuestionRaw``.

La interfaz se mantiene independiente del motor: :class:`OCRBackend` define el
contrato y :class:`OCRClient` es la implementación sobre el SDK oficial de
``openai``. Otros motores (por ejemplo GOT-OCR2.0) pueden exponer el mismo
contrato mediante un adaptador ligero.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from openai import OpenAI, OpenAIError

from taygeta.core.schema import ILLEGIBLE_MARKER
from taygeta.core.settings import Settings, get_settings

logger = logging.getLogger(__name__)

__all__ = [
    "ILLEGIBLE_MARKER",
    "OCRClient",
    "OCRError",
    "OCRImage",
    "OCRResult",
    "OCR_SYSTEM_PROMPT",
    "build_ocr_client",
    "sniff_mime_type",
    "to_data_uri",
]

#: Marcador explícito para segmentos que el OCR no puede leer con seguridad.
ILLEGIBLE = ILLEGIBLE_MARKER

#: Reintentos acotados por defecto ante errores transitorios del servicio.
DEFAULT_MAX_RETRIES = 2

#: Número máximo de tokens de la respuesta.
DEFAULT_MAX_TOKENS = 4096

#: Marcadores de URL aceptados directamente como entrada de imagen.
_URL_PREFIXES = ("http://", "https://", "data:")

OCR_SYSTEM_PROMPT = """Eres un transcriptor OCR de documentos científicos.

Tu única tarea es transcribir FIELMENTE el contenido visible de las imágenes a \
Markdown, sin interpretar ni resolver nada. Reglas obligatorias:

1. Conserva la estructura de lectura, la numeración, el texto del enunciado y \
las alternativas exactamente como aparecen.
2. Escribe las expresiones matemáticas con delimitadores TeX: $...$ para inline \
y $$...$$ para display.
3. Conserva funciones por partes, matrices, sumatorias, subíndices, \
superíndices, fracciones, límites, integrales y operadores definidos por el autor.
4. Usa los entornos TeX adecuados cuando correspondan a la estructura visible: \
cases, pmatrix, aligned, substack.
5. Mantén los placeholders de figura {{fig:N}} EXACTAMENTE en su posición dentro \
del texto; no los muevas ni los reescribas.
6. No inventes texto ilegible: escribe [ILEGIBLE] en su lugar y añade una línea \
final "ADVERTENCIAS: ..." indicando qué no pudiste leer.
7. NO resuelvas el ejercicio, NO añadas soluciones, respuestas correctas, \
claves de respuesta, explicaciones ni interpretaciones matemáticas.
8. Devuelve únicamente la transcripción en Markdown/TeX, sin comentarios sobre \
tu proceso.
"""


class OCRError(RuntimeError):
    """Error controlado del cliente OCR local."""


# ---------------------------------------------------------------------------
# Tipos de entrada/salida
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OCRImage:
    """Imagen de entrada al OCR: bytes crudos o una URL/data URI accesible."""

    data: bytes | None = None
    url: str | None = None
    mime_type: str = "image/png"
    page_index: int | None = None

    def __post_init__(self) -> None:
        if (self.data is None) == (self.url is None):
            raise OCRError("OCRImage requiere exactamente uno de: data o url")


@dataclass(slots=True)
class OCRResult:
    """Transcripción cruda devuelta por el OCR local."""

    text: str
    model: str
    pages: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    usage: dict[str, Any] | None = None

    @property
    def has_illegible(self) -> bool:
        """Indica si la transcripción contiene segmentos marcados como ilegibles."""
        return ILLEGIBLE in self.text

    @property
    def illegible_count(self) -> int:
        """Número de marcas ``[ILEGIBLE]`` presentes en la transcripción."""
        return self.text.count(ILLEGIBLE)


@runtime_checkable
class OCRBackend(Protocol):
    """Contrato mínimo e independiente del motor de OCR."""

    def transcribe(
        self,
        images: Sequence[OCRImage | bytes | str | Path],
        *,
        hints: str | None = None,
        source_reference: str | None = None,
    ) -> OCRResult:
        """Transcribe las imágenes a Markdown/TeX."""
        ...

    def health(self) -> bool:
        """Comprueba que el servicio OCR está disponible."""
        ...


# ---------------------------------------------------------------------------
# Utilidades de imagen
# ---------------------------------------------------------------------------


def sniff_mime_type(data: bytes) -> str:
    """Detecta el tipo MIME a partir de los bytes mágicos de la imagen."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"%PDF"):
        return "application/pdf"
    return "application/octet-stream"


def to_data_uri(data: bytes, mime_type: str | None = None) -> str:
    """Codifica bytes como data URI, admitido por motores que no descargan URLs."""
    resolved = mime_type or sniff_mime_type(data)
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{resolved};base64,{encoded}"


# ---------------------------------------------------------------------------
# Cliente OCR (OpenAI-compatible)
# ---------------------------------------------------------------------------


class OCRClient:
    """Cliente del servicio OCR local compatible con la API de OpenAI."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        """Crea el cliente OCR.

        Args:
            settings: Configuración a usar; por defecto la global.
            client: Cliente compatible con el SDK de OpenAI (inyectable en
                pruebas unitarias). Si se omite, se construye uno real.
            model / base_url / api_key / timeout: Sobrescriben los valores de
                configuración, útil para seleccionar el motor sin tocar código.
        """
        self.settings = settings or get_settings()
        self._client = client
        self._model = model
        self._base_url = (base_url or self.settings.OCR_BASE_URL).rstrip("/")
        self._api_key = api_key if api_key is not None else self.settings.OCR_API_KEY
        self._timeout = timeout if timeout is not None else float(self.settings.OCR_TIMEOUT_SECONDS)
        self._max_retries = max(0, max_retries)
        self._max_tokens = max_tokens

    # -- Configuración ------------------------------------------------------
    @property
    def model(self) -> str:
        """Identificador del modelo OCR configurado."""
        return self._model or self.settings.OCR_MODEL

    @property
    def base_url(self) -> str:
        """Endpoint OpenAI-compatible del servicio OCR."""
        return self._base_url

    @property
    def client(self) -> Any:
        """Cliente ``openai`` construido de forma perezosa."""
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> OpenAI:
        logger.info("cliente OCR -> base_url=%s model=%s", self._base_url, self.model)
        return OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

    # -- Salud --------------------------------------------------------------
    def health(self) -> bool:
        """Comprueba el endpoint de salud del servicio OCR (lista de modelos)."""
        try:
            self.client.models.list()
            return True
        except Exception as exc:  # noqa: BLE001 - cualquier fallo implica no saludable
            logger.warning("el servicio OCR no está disponible en %s: %s", self._base_url, exc)
            return False

    is_healthy = health

    # -- Transcripción ------------------------------------------------------
    def transcribe(
        self,
        images: Sequence[OCRImage | bytes | str | Path],
        *,
        hints: str | None = None,
        source_reference: str | None = None,
    ) -> OCRResult:
        """Transcribe una o varias imágenes a Markdown con matemáticas TeX.

        Args:
            images: Imágenes de entrada (bytes, rutas, URLs o :class:`OCRImage`).
            hints: Contexto correctivo opcional aportado por el usuario.
            source_reference: Referencia de origen para trazabilidad.

        Raises:
            OCRError: si no hay imágenes, si el proveedor falla o la respuesta
                no es interpretable.
        """
        if not images:
            raise OCRError("se requiere al menos una imagen para transcribir")

        content: list[dict[str, Any]] = [
            {"type": "text", "text": self._build_instruction(hints, source_reference, images)}
        ]
        content.extend(self._to_image_part(image) for image in images)

        messages = [
            {"role": "system", "content": OCR_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=self._max_tokens,
            )
        except OpenAIError as exc:
            raise OCRError(f"fallo del servicio OCR ({self.base_url}): {exc}") from exc
        except Exception as exc:  # cliente inyectado u otros motores
            raise OCRError(f"error inesperado del servicio OCR: {exc}") from exc

        text = self._extract_text(response)
        pages = self._collect_pages(images)
        return OCRResult(
            text=text,
            model=self.model,
            pages=pages,
            warnings=self._collect_warnings(text),
            usage=self._extract_usage(response),
        )

    def transcribe_page(
        self,
        image: OCRImage | bytes | str | Path,
        *,
        page_index: int | None = None,
        hints: str | None = None,
        source_reference: str | None = None,
    ) -> OCRResult:
        """Transcribe una única página o recorte."""
        entry: OCRImage | bytes | str | Path
        if isinstance(image, (bytes, bytearray)):
            payload = bytes(image)
            entry = OCRImage(
                data=payload, mime_type=sniff_mime_type(payload), page_index=page_index
            )
        elif isinstance(image, OCRImage) and page_index is not None:
            entry = OCRImage(
                data=image.data, url=image.url, mime_type=image.mime_type, page_index=page_index
            )
        else:
            entry = image
        return self.transcribe([entry], hints=hints, source_reference=source_reference)

    # -- Internos -----------------------------------------------------------
    def _build_instruction(
        self,
        hints: str | None,
        source_reference: str | None,
        images: Sequence[Any],
    ) -> str:
        lines = ["Transcribe fielmente a Markdown el contenido visible de la(s) imagen(es)."]
        pages = self._collect_pages(images)
        if source_reference:
            lines.append(f"Origen: {source_reference}")
        if pages:
            lines.append("Páginas (índice físico basado en cero): " + ", ".join(map(str, pages)))
        if hints:
            lines.append(
                "Contexto adicional del revisor (úsalo solo para desambiguar la lectura, "
                f"nunca para inventar contenido): {hints}"
            )
        lines.append(
            "Recuerda: no resuelvas el ejercicio, no añadas soluciones ni respuestas correctas."
        )
        return "\n".join(lines)

    def _coerce_image(self, image: OCRImage | bytes | str | Path) -> OCRImage:
        if isinstance(image, OCRImage):
            return image
        if isinstance(image, (bytes, bytearray)):
            payload = bytes(image)
            return OCRImage(data=payload, mime_type=sniff_mime_type(payload))
        if isinstance(image, Path):
            return self._image_from_path(image)
        if isinstance(image, str):
            if image.startswith(_URL_PREFIXES):
                return OCRImage(url=image)
            return self._image_from_path(Path(image))
        raise OCRError(f"tipo de imagen no admitido: {type(image).__name__}")

    @staticmethod
    def _image_from_path(path: Path) -> OCRImage:
        if not path.is_file():
            raise OCRError(f"el archivo de imagen no existe: {path}")
        payload = path.read_bytes()
        return OCRImage(data=payload, mime_type=sniff_mime_type(payload))

    def _to_image_part(self, image: OCRImage | bytes | str | Path) -> dict[str, Any]:
        coerced = self._coerce_image(image)
        url = coerced.url
        if url is None:
            if coerced.data is None:
                raise OCRError("la imagen no contiene datos ni URL")
            url = to_data_uri(coerced.data, coerced.mime_type)
        return {"type": "image_url", "image_url": {"url": url}}

    @staticmethod
    def _collect_pages(images: Iterable[Any]) -> list[int]:
        pages: list[int] = []
        for image in images:
            page_index = getattr(image, "page_index", None)
            if isinstance(page_index, int) and page_index not in pages:
                pages.append(page_index)
        return pages

    @staticmethod
    def _extract_text(response: Any) -> str:
        try:
            choices = response.choices
        except AttributeError as exc:
            raise OCRError("respuesta del OCR sin campo 'choices'") from exc
        if not choices:
            raise OCRError("la respuesta del OCR no contiene alternativas")
        content = getattr(choices[0].message, "content", None)
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                if text:
                    parts.append(str(text))
            content = "".join(parts)
        return (content or "").strip()

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, Any] | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        if isinstance(usage, dict):
            return dict(usage)
        return None

    @staticmethod
    def _collect_warnings(text: str) -> list[str]:
        warnings: list[str] = []
        count = text.count(ILLEGIBLE)
        if count:
            warnings.append(
                f"se detectaron {count} segmento(s) marcados como {ILLEGIBLE}; "
                "requieren revisión humana"
            )
        if not text.strip():
            warnings.append("la transcripción del OCR quedó vacía")
        return warnings


def build_ocr_client(settings: Settings | None = None) -> OCRClient:
    """Construye el cliente OCR con la configuración global ``OCR_*``."""
    return OCRClient(settings)
