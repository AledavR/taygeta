"""Esquemas Pydantic v2 canónicos del proyecto Taygeta.

Este módulo define los modelos de datos empleados por la ingesta híbrida, la
clasificación curricular y el almacenamiento de preguntas.

Reglas transversales:

* ``extra="forbid"`` en todos los modelos: cualquier campo no declarado, en
  particular cualquier intento de añadir soluciones, claves de respuesta o
  respuestas correctas, provoca un error de validación.
* Los slugs curriculares (``course``, ``topic``, ``subtopic`` y prerrequisitos)
  se validan con la expresión regular ``^[a-z0-9_]+$`` (sin espacios).
* Los enunciados usan Markdown con matemáticas TeX ``$...$`` y ``$$...$$`` y
  placeholders de figura ``{{fig:N}}``. Los delimitadores LaTeX no canónicos
  (``\\[...\\]`` y ``\\(...\\)``) se normalizan automáticamente con
  :func:`sanitize_math_delimiters` al validar los esquemas.

NOTA CRÍTICA: Ningún modelo admite ni almacena soluciones, claves de respuestas,
respuestas correctas ni razonamientos que resuelvan el ejercicio. La extracción
transcribe únicamente el contenido visible.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

__all__ = [
    "CURRICULAR_SLUG_PATTERN",
    "ILLEGIBLE_MARKER",
    "PLACEHOLDER_PATTERN",
    "RELATIVE_ASSET_PREFIX",
    "Choice",
    "EnrichedQuestion",
    "FigureReference",
    "QuestionRaw",
    "Slug",
    "TranscriptionIntermediate",
    "compute_canonical_hash",
    "deterministic_question_id",
    "get_public_url",
    "is_relative_asset_path",
    "new_question_id",
    "normalize_slug",
    "sanitize_math_delimiters",
    "slugify",
    "utc_now_iso",
]


# ---------------------------------------------------------------------------
# Constantes y tipos auxiliares
# ---------------------------------------------------------------------------

#: Expresión regular de slugs curriculares: minúsculas, dígitos y guion bajo.
CURRICULAR_SLUG_PATTERN = r"^[a-z0-9_]+$"

#: Convención explícita para marcar fragmentos ilegibles durante el OCR.
ILLEGIBLE_MARKER = "[ILEGIBLE]"

#: Placeholders que intercalan figuras dentro del enunciado, p. ej. ``{{fig:1}}``.
PLACEHOLDER_PATTERN = r"\{\{fig:(\d+)\}\}"

#: Prefijo canónico de las rutas relativas de assets, p. ej.
#: ``/buckets/exams/img.png`` (equivalente a ``s3://exams/img.png``).
RELATIVE_ASSET_PREFIX = "/buckets/"

_PLACEHOLDER_RE = re.compile(PLACEHOLDER_PATTERN)
_FIGURE_ID_PATTERN = r"^fig:\d+$"

#: Delimitadores TeX no estándar que emiten algunos modelos OCR/LLM y que rompen
#: el renderizado de KaTeX/Typst: ``\[...\]`` denota display y ``\(...\)`` denota
#: inline. Se toleran barras invertidas escapadas de más (``\\[``, ``\\]``).
_DISPLAY_MATH_RE = re.compile(r"\\+\[(.*?)\\+\]", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"\\+\((.*?)\\+\)", re.DOTALL)

#: Slug curricular validado (minúsculas, dígitos y guion bajo, sin espacios).
Slug = Annotated[
    str,
    StringConstraints(pattern=CURRICULAR_SLUG_PATTERN, strip_whitespace=False),
]

#: Namespace estable para UUID5 deterministas de preguntas.
_QUESTION_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def slugify(value: str | None) -> str:
    """Convierte texto libre en un slug válido ``^[a-z0-9_]+$`` (o ``""``).

    - separa las tildes/diacríticos (``Cálculo`` → ``calculo``);
    - pasa a minúsculas;
    - sustituye espacios y caracteres especiales por ``_``;
    - colapsa ``_`` repetidos y recorta los de los extremos.
    """
    if value is None:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(value))
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    normalized = re.sub(r"[^a-z0-9_]+", "_", without_marks.strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_")


def normalize_slug(value: str | None, default: str = "general") -> str:
    """Normaliza un slug y devuelve ``default`` cuando queda vacío."""
    return slugify(value) or default


def sanitize_math_delimiters(text: str) -> str:
    """Normaliza delimitadores matemáticos TeX a ``$...$`` / ``$$...$$``.

    Algunos modelos OCR/LLM emiten delimitadores LaTeX de estilo ``\\[...\\]``
    (display) y ``\\(...\\)`` (inline), que KaTeX y Typst no interpretan igual que
    los delimitadores canónicos del proyecto. Esta función los reescribe a
    ``$$...$$`` y ``$...$`` respectivamente y tolera barras invertidas escapadas
    de más (por ejemplo ``\\\\[...\\\\]``). Los enunciados sin esos patrones se
    devuelven intactos.
    """
    if not text:
        return text
    sanitized = _DISPLAY_MATH_RE.sub(lambda match: f"$${match.group(1)}$$", text)
    sanitized = _INLINE_MATH_RE.sub(lambda match: f"${match.group(1)}$", sanitized)
    return sanitized


def is_relative_asset_path(value: str | None) -> bool:
    """Indica si ``value`` es una ruta relativa de asset ``/buckets/{bucket}/{key}``.

    Rechaza segmentos vacíos, ``.`` y ``..`` para impedir traversal, pero acepta
    el prefijo ``/buckets/`` (que, a diferencia de ``/app/...`` o ``file://``, no
    es una ruta local absoluta sino una ruta relativa al almacenamiento público).
    """
    if not value or not value.startswith(RELATIVE_ASSET_PREFIX):
        return False
    rest = value[len(RELATIVE_ASSET_PREFIX) :]
    if not rest:
        return False
    return all(segment not in ("", ".", "..") for segment in rest.split("/"))


def get_public_url(relative_path: str) -> str:
    """Compone la URL pública concatenando ``S3_PUBLIC_BASE_URL`` con la ruta relativa.

    Las imágenes se almacenan en la base de datos como rutas relativas
    ``/buckets/{bucket}/{key}``; esta función compone la URL completa justo en el
    límite de salida. Si ``relative_path`` está vacío, devuelve ``""``.
    """
    if not relative_path:
        return ""
    from taygeta.core.settings import get_settings

    base = (get_settings().S3_PUBLIC_BASE_URL or "").rstrip("/")
    path = relative_path if relative_path.startswith("/") else "/" + relative_path
    return base + path


def compute_canonical_hash(tex_string: str) -> str:
    """Calcula un hash canónico estable para la deduplicación de preguntas.

    Limpia los espacios y saltos de línea del enunciado (colapsando cualquier
    secuencia de espacios en blanco a un único espacio) y devuelve el hash
    SHA-256 hexadecimal del contenido normalizado. No usa soluciones ni
    respuestas correctas: solo el contenido visible del enunciado.
    """
    normalized = " ".join(str(tex_string).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def new_question_id() -> str:
    """Genera un UUID aleatorio (versión 4) como identificador de pregunta."""
    return str(uuid.uuid4())


def deterministic_question_id(*parts: str) -> str:
    """Genera un UUID5 determinista a partir de las partes indicadas.

    Permite que una misma pregunta obtenga siempre el mismo identificador
    mientras su contenido de origen no cambie.
    """
    if not parts:
        raise ValueError("se requiere al menos una parte para el UUID determinista")
    return str(uuid.uuid5(_QUESTION_ID_NAMESPACE, "|".join(parts)))


def utc_now_iso() -> str:
    """Devuelve la fecha/hora actual en ISO 8601 con zona horaria UTC."""
    return datetime.now(tz=UTC).isoformat()


# ---------------------------------------------------------------------------
# 3.1. FigureReference
# ---------------------------------------------------------------------------


class FigureReference(BaseModel):
    """Referencia geométrica a una figura extraída de un documento.

    ``bbox`` usa el formato ``[ymin, xmin, ymax, xmax]`` con coordenadas
    normalizadas en el rango ``0-1000``. ``asset_path`` es un identificador
    estable de objeto:

    * una ruta relativa pública ``/buckets/{bucket}/{key}`` (forma canónica
      recomendada, p. ej. ``/buckets/exams/img.png``);
    * una URI ``s3://bucket/key``;
    * una clave de objeto relativa al bucket (sin ``/`` inicial).

    Nunca una ruta local absoluta como ``/app/assets/...``.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(
        ...,
        pattern=_FIGURE_ID_PATTERN,
        description='Identificador estable de la figura, por ejemplo "fig:1".',
    )
    page: int = Field(
        ...,
        ge=0,
        description="Índice físico de la página basado en cero.",
    )
    bbox: list[int] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box [ymin, xmin, ymax, xmax] normalizado a 0-1000.",
    )
    caption: str | None = Field(
        default=None,
        description="Pie de figura visible, si existe.",
    )
    asset_path: str | None = Field(
        default=None,
        description=(
            "Identificador de objeto estable: ruta relativa /buckets/{bucket}/{key}, "
            "s3://bucket/key o clave relativa. Nunca una ruta local absoluta."
        ),
    )

    @field_validator("bbox")
    @classmethod
    def _validate_bbox(cls, value: list[int]) -> list[int]:
        ymin, xmin, ymax, xmax = value
        if any(coord < 0 or coord > 1000 for coord in value):
            raise ValueError("bbox debe tener coordenadas normalizadas entre 0 y 1000")
        if ymin >= ymax or xmin >= xmax:
            raise ValueError(
                "bbox debe cumplir ymin < ymax y xmin < xmax "
                f"(recibido [ymin={ymin}, xmin={xmin}, ymax={ymax}, xmax={xmax}])"
            )
        return value

    @field_validator("asset_path")
    @classmethod
    def _validate_asset_path(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if value.startswith(RELATIVE_ASSET_PREFIX):
            if not is_relative_asset_path(value):
                raise ValueError(
                    "la ruta relativa de asset debe tener la forma /buckets/{bucket}/{key} "
                    "sin segmentos vacíos, '.' ni '..'"
                )
            return value
        if value.startswith("/") or value.startswith("file://"):
            raise ValueError(
                "asset_path no puede ser una ruta local absoluta; usa /buckets/{bucket}/{key}, "
                "s3://bucket/key o una clave de objeto relativa al bucket"
            )
        if re.match(r"^[a-zA-Z]:[\\/]", value):
            raise ValueError(
                "asset_path no puede ser una ruta local absoluta de Windows; usa "
                "/buckets/{bucket}/{key}, s3://bucket/key o una clave de objeto relativa al bucket"
            )
        return value


# ---------------------------------------------------------------------------
# 3.2. Choice
# ---------------------------------------------------------------------------


class Choice(BaseModel):
    """Alternativa de una pregunta de opción múltiple."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        min_length=1,
        pattern=r"^[A-Za-z0-9]+$",
        description='Identificador de la alternativa, por ejemplo "A", "B", "C" o "D".',
    )
    content_raw: str = Field(
        ...,
        min_length=1,
        description="Contenido Markdown de la alternativa con expresiones matemáticas TeX.",
    )

    @field_validator("content_raw", mode="before")
    @classmethod
    def _normalize_content_math(cls, value: Any) -> Any:
        """Normaliza los delimitadores matemáticos antes de validar la alternativa."""
        return sanitize_math_delimiters(value) if isinstance(value, str) else value


# ---------------------------------------------------------------------------
# 3.3. QuestionRaw
# ---------------------------------------------------------------------------


class QuestionRaw(BaseModel):
    """Pregunta transcrita y estructurada, sin solución ni respuesta correcta.

    El enunciado usa Markdown con ``$...$`` y ``$$...$$`` para matemáticas y
    placeholders ``{{fig:N}}`` para intercalar figuras en el punto exacto del
    flujo de lectura. Los placeholders y las figuras declaradas deben coincidir
    uno a uno.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description="UUID determinista o generado que identifica la pregunta.",
    )
    statement_raw: str = Field(
        ...,
        min_length=1,
        description=(
            "Enunciado en Markdown con matemáticas $...$ / $$...$$ y placeholders "
            "{{fig:N}} para figuras."
        ),
    )
    question_type: Literal["multiple_choice", "open_ended", "true_false", "numerical"] = Field(
        ...,
        description="Tipo de pregunta según su formato visible.",
    )
    choices: list[Choice] | None = Field(
        default=None,
        description="Alternativas; obligatorias (>= 2) para multiple_choice.",
    )
    figures: list[FigureReference] = Field(
        default_factory=list,
        description="Figuras referenciadas por el enunciado mediante {{fig:N}}.",
    )
    notation_notes: list[str] = Field(
        default_factory=list,
        description="Notas sobre ambigüedades o incertidumbres de notación detectadas.",
    )

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        try:
            uuid.UUID(value)
        except (ValueError, AttributeError, TypeError) as exc:
            raise ValueError("id debe ser un UUID válido") from exc
        return value

    @field_validator("statement_raw", mode="before")
    @classmethod
    def _normalize_statement_math(cls, value: Any) -> Any:
        """Sanitiza los delimitadores matemáticos justo al validar/ingerir.

        Se aplica antes de la validación del modelo y, por tanto, antes de
        cualquier persistencia: todo ``statement_raw`` que entra al sistema
        (ingesta, refinamiento, UI o base de datos) queda con ``$...$`` y
        ``$$...$$``.
        """
        return sanitize_math_delimiters(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _validate_structure(self) -> QuestionRaw:
        figure_ids = [figure.id for figure in self.figures]
        if len(figure_ids) != len(set(figure_ids)):
            raise ValueError("los identificadores de figuras deben ser únicos")

        declared = set(figure_ids)
        referenced = {f"fig:{match}" for match in _PLACEHOLDER_RE.findall(self.statement_raw)}
        missing = sorted(referenced - declared)
        unused = sorted(declared - referenced)
        if missing:
            raise ValueError(
                "el enunciado referencia placeholders sin figura declarada: " + ", ".join(missing)
            )
        if unused:
            raise ValueError(
                "hay figuras declaradas sin placeholder en el enunciado: " + ", ".join(unused)
            )

        if self.question_type == "multiple_choice":
            choices = self.choices or []
            if len(choices) < 2:
                raise ValueError(
                    "las preguntas multiple_choice requieren al menos dos alternativas"
                )
        if self.choices is not None:
            choice_ids = [choice.id for choice in self.choices]
            if len(choice_ids) != len(set(choice_ids)):
                raise ValueError("los identificadores de alternativas deben ser únicos")
        return self


# ---------------------------------------------------------------------------
# 3.4. EnrichedQuestion
# ---------------------------------------------------------------------------


class EnrichedQuestion(QuestionRaw):
    """Pregunta raw enriquecida con clasificación curricular y metadatos."""

    model_config = ConfigDict(extra="forbid")

    course: Slug = Field(
        ...,
        description="Slug del curso, sin espacios (regex ^[a-z0-9_]+$).",
    )
    topic: Slug = Field(
        ...,
        description="Slug del tema, sin espacios (regex ^[a-z0-9_]+$).",
    )
    subtopic: Slug | None = Field(
        default=None,
        description="Slug del subtema, sin espacios, o None (regex ^[a-z0-9_]+$).",
    )
    difficulty: int = Field(
        ...,
        ge=1,
        le=5,
        description="Dificultad curricular en el rango 1-5.",
    )
    prerequisites: list[Slug] = Field(
        default_factory=list,
        description="Slugs de los prerrequisitos (formato ^[a-z0-9_]+$).",
    )
    source_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadatos del documento de origen (páginas, fichero, etc.).",
    )
    created_at: str = Field(
        ...,
        description="Fecha/hora de creación en ISO 8601, preferentemente con zona horaria.",
    )
    canonical_hash: str | None = Field(
        default=None,
        description=(
            "Hash canónico (hex SHA-256) del enunciado normalizado, para la "
            "deduplicación futura. Se calcula con ``compute_canonical_hash`` y nunca "
            "incluye soluciones ni respuestas correctas."
        ),
    )

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: str) -> str:
        candidate = value.strip()
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise ValueError(
                f"created_at debe ser una fecha/hora ISO 8601 válida (recibido {value!r})"
            ) from exc
        return value

    @field_validator("prerequisites", mode="before")
    @classmethod
    def _normalize_prerequisites(cls, value: Any) -> list[str]:
        """Normaliza y deduplica los prerrequisitos antes de validarlos.

        Tolera una cadena suelta (separada por comas o punto y coma) y aplica
        ``slugify`` a cada elemento, descartando vacíos y duplicados. La
        canonicalización semántica (reutilización de slugs ya existentes en la
        base de datos) la realiza el LLM durante la estructuración mediante el
        catálogo dinámico inyectado en el prompt; aquí solo se garantiza el
        formato básico ``^[a-z0-9_]+$`` impuesto por :data:`Slug`.
        """
        if value is None:
            return []
        if isinstance(value, str):
            value = value.replace(";", ",").split(",")
        if not isinstance(value, (list, tuple, set)):
            return []
        result: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            slug = slugify(item)
            if slug and slug not in result:
                result.append(slug)
        return result


# ---------------------------------------------------------------------------
# 3.5. Esquema interno de transcripción
# ---------------------------------------------------------------------------


class TranscriptionIntermediate(BaseModel):
    """Salida cruda del OCR local antes del pase remoto de estructuración.

    Conserva el texto Markdown/TeX bruto, la referencia de origen, las páginas
    involucradas, la información de figuras y las advertencias de lectura. NO
    sustituye a :class:`QuestionRaw`: no realiza clasificación curricular ni
    validación lógica.
    """

    model_config = ConfigDict(extra="forbid")

    source_reference: str = Field(
        ...,
        min_length=1,
        description="Referencia estable del origen (nombre de fichero o identificador).",
    )
    raw_markdown: str = Field(
        ...,
        description="Texto Markdown/TeX bruto transcrito por el OCR local.",
    )
    pages: list[int] = Field(
        default_factory=list,
        description="Índices físicos de página involucrados, basados en cero.",
    )
    figures: list[FigureReference] = Field(
        default_factory=list,
        description="Figuras detectadas con su geometría y referencia de origen.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Advertencias de lectura, incluidos segmentos marcados como ilegibles.",
    )

    @field_validator("pages")
    @classmethod
    def _validate_pages(cls, value: list[int]) -> list[int]:
        if any(page < 0 for page in value):
            raise ValueError("los índices de página deben ser mayores o iguales a cero")
        if len(value) != len(set(value)):
            raise ValueError("los índices de página deben ser únicos")
        return value
