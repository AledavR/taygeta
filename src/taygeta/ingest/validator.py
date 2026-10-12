"""Validación en dos capas del pipeline de ingesta (sección 4.4).

* **Capa 1 — heurística local** (:func:`validate_local`): comprobaciones baratas
  (objetivo < 1 ms en documentos pequeños) que identifican errores probables sin
  reescribir enunciados: longitud mínima, paridad de delimitadores matemáticos,
  balance de llaves, concordancia placeholder/figure, validez de figuras y
  páginas, campos obligatorios vacíos y alternativas duplicadas.
* **Capa 2 — validación semántica remota** (:class:`SemanticValidator`): usa el
  LLM remoto configurado con ``LLM_*`` (independiente del OCR) para validar
  legibilidad, suficiencia, truncamientos, alternativas incompletas y
  contradicciones obvias, *sin resolver* el ejercicio. Devuelve
  :class:`LogicalValidationResult`.

Los objetos que fallan cualquier validación quedan en staging con estado
``NEEDS_ATTENTION`` y un motivo técnico; los que pasan pueden marcarse ``VALID``
pero no se guardan definitivamente hasta la confirmación humana.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from taygeta.core.schema import PLACEHOLDER_PATTERN, QuestionRaw
from taygeta.core.settings import Settings, get_settings

if TYPE_CHECKING:
    import instructor  # noqa: F401

__all__ = [
    "SEVERITY_ERROR",
    "SEVERITY_WARNING",
    "STATUS_VALID",
    "STATUS_NEEDS_ATTENTION",
    "MIN_STATEMENT_LENGTH",
    "LocalIssue",
    "LocalValidationResult",
    "LogicalValidationResult",
    "RemoteValidationError",
    "SemanticValidator",
    "validate_local",
]

#: Estados de staging admitidos para las preguntas.
STATUS_VALID = "VALID"
STATUS_NEEDS_ATTENTION = "NEEDS_ATTENTION"

#: Niveles de severidad de una incidencia local.
SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"

#: Longitud mínima por defecto del enunciado (sección 4.4).
MIN_STATEMENT_LENGTH = 20

_PLACEHOLDER_RE = re.compile(PLACEHOLDER_PATTERN)
_ALLOWED_TYPES = {"multiple_choice", "open_ended", "true_false", "numerical"}


@dataclass(frozen=True, slots=True)
class LocalIssue:
    """Incidencia detectada por la heurística local."""

    code: str
    message: str
    severity: Literal["error", "warning"] = SEVERITY_ERROR


@dataclass(slots=True)
class LocalValidationResult:
    """Resultado de la validación local (Capa 1)."""

    is_valid: bool
    issues: list[LocalIssue] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def errors(self) -> list[LocalIssue]:
        return [issue for issue in self.issues if issue.severity == SEVERITY_ERROR]

    @property
    def warnings(self) -> list[LocalIssue]:
        return [issue for issue in self.issues if issue.severity == SEVERITY_WARNING]

    def format_reasons(self) -> str:
        """Une todas las incidencias en un motivo técnico legible."""
        if not self.issues:
            return ""
        return "; ".join(f"{issue.code}: {issue.message}" for issue in self.issues)


# ---------------------------------------------------------------------------
# Capa 1 — Heurística local
# ---------------------------------------------------------------------------


def _check_min_length(statement: str, min_length: int) -> LocalIssue | None:
    if len(statement.strip()) < min_length:
        return LocalIssue(
            code="statement_too_short",
            message=(
                f"enunciado de {len(statement.strip())} caracteres (< {min_length}); "
                "puede ser válido si es una expresión breve justificada"
            ),
            severity=SEVERITY_WARNING,
        )
    return None


def _scan_math_delimiters(text: str) -> str:
    """Devuelve una descripción del desbalance de delimitadores matemáticos, o ''."""
    display_open = False
    inline_open = False
    index = 0
    length = len(text)
    while index < length:
        if text[index : index + 2] == "$$":
            display_open = not display_open
            index += 2
            continue
        if text[index] == "$":
            if index > 0 and text[index - 1] == "\\":
                index += 1
                continue
            inline_open = not inline_open
            index += 1
            continue
        index += 1
    problems: list[str] = []
    if display_open:
        problems.append("delimitador de display $$ sin cerrar")
    if inline_open:
        problems.append("delimitador inline $ sin cerrar")
    return "; ".join(problems)


def _check_math_delimiters(statement: str, choices_text: str) -> LocalIssue | None:
    combined = statement + "\n" + choices_text
    problem = _scan_math_delimiters(combined)
    if problem:
        return LocalIssue(code="unbalanced_math_delimiters", message=problem)
    return None


def _check_braces(statement: str, choices_text: str) -> LocalIssue | None:
    depth = 0
    text = statement + "\n" + choices_text
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char == "\\" and index + 1 < length:
            # Saltar el carácter escapado (\{ \} \frac ...) para no confundir llaves.
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        index += 1
    if depth != 0:
        return LocalIssue(
            code="unbalanced_braces",
            message=f"llaves {{}} desbalanceadas (saldo {depth:+d})",
        )
    return None


def _check_placeholder_figure_match(
    statement: str, choices_text: str, figure_ids: set[str]
) -> LocalIssue | None:
    referenced = {
        f"fig:{match}" for match in _PLACEHOLDER_RE.findall(statement + "\n" + choices_text)
    }
    missing = sorted(referenced - figure_ids)
    unused = sorted(figure_ids - referenced)
    if missing or unused:
        message = []
        if missing:
            message.append("placeholders sin figura declarada: " + ", ".join(missing))
        if unused:
            message.append("figuras declaradas sin placeholder: " + ", ".join(unused))
        return LocalIssue(code="placeholder_figure_mismatch", message="; ".join(message))
    return None


def _check_figure_references(question: QuestionRaw, max_pages: int | None) -> list[LocalIssue]:
    issues: list[LocalIssue] = []
    seen: set[str] = set()
    for figure in question.figures:
        if figure.id in seen:
            issues.append(
                LocalIssue(code="duplicate_figure_id", message=f"figura duplicada {figure.id}")
            )
        seen.add(figure.id)
        if figure.page < 0:
            issues.append(
                LocalIssue(code="invalid_figure_page", message=f"página negativa en {figure.id}")
            )
        if max_pages is not None and figure.page >= max_pages:
            issues.append(
                LocalIssue(
                    code="figure_page_out_of_range",
                    message=f"página {figure.page} de {figure.id} fuera de rango (< {max_pages})",
                )
            )
        ymin, xmin, ymax, xmax = figure.bbox
        if not (0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000):
            issues.append(
                LocalIssue(code="invalid_figure_bbox", message=f"BBox inválida en {figure.id}")
            )
    return issues


def _check_required_fields(question: QuestionRaw) -> list[LocalIssue]:
    issues: list[LocalIssue] = []
    if not question.statement_raw.strip():
        issues.append(LocalIssue(code="empty_statement", message="enunciado vacío"))
    if question.question_type not in _ALLOWED_TYPES:
        issues.append(
            LocalIssue(
                code="invalid_question_type",
                message=f"tipo de pregunta inválido: {question.question_type!r}",
            )
        )
    if question.question_type == "multiple_choice":
        if not question.choices or len(question.choices) < 2:
            issues.append(
                LocalIssue(
                    code="missing_choices",
                    message="multiple_choice requiere al menos dos alternativas",
                )
            )
    if question.choices is not None:
        for choice in question.choices:
            if not choice.content_raw.strip():
                issues.append(
                    LocalIssue(
                        code="empty_choice_content",
                        message=f"contenido vacío en la alternativa {choice.id}",
                    )
                )
    return issues


def _check_duplicate_choices(question: QuestionRaw) -> LocalIssue | None:
    if question.choices is None:
        return None
    ids = [choice.id for choice in question.choices]
    duplicates = sorted({choice_id for choice_id in ids if ids.count(choice_id) > 1})
    if duplicates:
        return LocalIssue(
            code="duplicate_choices", message="alternativas duplicadas: " + ", ".join(duplicates)
        )
    return None


def validate_local(
    question: QuestionRaw,
    *,
    min_statement_length: int = MIN_STATEMENT_LENGTH,
    max_pages: int | None = None,
) -> LocalValidationResult:
    """Ejecuta la Capa 1 (heurística local) sobre una pregunta.

    Args:
        question: Pregunta ya validada por el esquema Pydantic.
        min_statement_length: Longitud mínima del enunciado (0 para desactivar).
        max_pages: Número total de páginas del documento para validar figuras.
    """
    issues: list[LocalIssue] = []
    checks: dict[str, bool] = {}

    choices_text = "\n".join(choice.content_raw for choice in (question.choices or []))
    figure_ids = {figure.id for figure in question.figures}

    def add(code: str, issue: LocalIssue | None) -> None:
        checks[code] = issue is None
        if issue is not None:
            issues.append(issue)

    if min_statement_length > 0:
        add("min_length", _check_min_length(question.statement_raw, min_statement_length))
    add("math_delimiters", _check_math_delimiters(question.statement_raw, choices_text))
    add("braces", _check_braces(question.statement_raw, choices_text))
    add(
        "placeholder_figure_match",
        _check_placeholder_figure_match(question.statement_raw, choices_text, figure_ids),
    )
    issues.extend(_check_figure_references(question, max_pages))
    checks["figure_references"] = not any(
        issue.code in {"duplicate_figure_id", "invalid_figure_page", "invalid_figure_bbox"}
        for issue in issues
    )
    issues.extend(_check_required_fields(question))
    checks["required_fields"] = not any(issue.code.endswith("empty") for issue in issues)
    add("duplicate_choices", _check_duplicate_choices(question))

    is_valid = not any(issue.severity == SEVERITY_ERROR for issue in issues)
    return LocalValidationResult(is_valid=is_valid, issues=issues, checks=checks)


# ---------------------------------------------------------------------------
# Capa 2 — Validación semántica remota
# ---------------------------------------------------------------------------


class LogicalValidationResult(BaseModel):
    """Resultado de la validación semántica remota (Capa 2)."""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    rejection_reason: str = Field(default="")


_SEMANTIC_SYSTEM_PROMPT = """Eres un revisor de calidad de enunciados de exámenes \
científicos. Evalúa SOLO la legibilidad y la suficiencia del enunciado. NO resuelvas \
el ejercicio, NO calcules su resultado y NO evalúes si la respuesta es correcta.

Considera el enunciado INVALIDO (is_valid=false) si:
- está truncado o incompleto;
- contiene alternativas incompletas o mal delimitadas;
- presenta contradicciones textuales obvias;
- no contiene información suficiente para entender qué se solicita;
- tiene problemas de estructura que impiden su lectura.

En caso de ser inválido, escribe un rejection_reason breve y concreto en español \
indicando el problema. Si es válido, deja rejection_reason vacío.
"""


class RemoteValidationError(RuntimeError):
    """Error controlado de la validación semántica remota."""


class SemanticValidator:
    """Capa 2: validación semántica mediante el LLM remoto (independiente del OCR)."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
        model: str | None = None,
        max_retries: int | None = None,
    ) -> None:
        """Crea el validador semántico.

        Args:
            settings: Configuración a usar; por defecto la global.
            client: Cliente ``instructor`` ya construido (inyectable en pruebas).
            model: Sobrescribe ``LLM_MODEL``.
        """
        self.settings = settings or get_settings()
        self._client = client
        self._model = model
        self._max_retries = (
            max_retries if max_retries is not None else self.settings.LLM_MAX_RETRIES
        )

    @property
    def model(self) -> str:
        return self._model or self.settings.LLM_MODEL

    @property
    def client(self) -> Any:
        if self._client is None:
            import instructor

            self._client = instructor.from_openai(self._openai_client())
        return self._client

    def _openai_client(self) -> OpenAI:
        return OpenAI(
            base_url=self.settings.LLM_BASE_URL,
            api_key=self.settings.LLM_API_KEY,
            timeout=float(self.settings.LLM_TIMEOUT_SECONDS),
            max_retries=self._max_retries,
        )

    def validate(
        self, question: QuestionRaw, *, hints: str | None = None
    ) -> LogicalValidationResult:
        """Evalúa semánticamente una pregunta.

        Raises:
            RemoteValidationError: si el proveedor falla o la respuesta no se
                puede validar.
        """
        user_text = "Enunciado a revisar (Markdown/TeX):\n\n" + question.statement_raw
        if question.choices:
            alternatives = "\n".join(
                f"{choice.id}: {choice.content_raw}" for choice in question.choices
            )
            user_text += f"\n\nAlternativas:\n{alternatives}"
        user_text += f"\n\nTipo: {question.question_type}"
        if hints:
            user_text += f"\n\nContexto adicional del revisor: {hints}"
        user_text += "\n\nEvalúa y devuelve is_valid y rejection_reason."

        try:
            result = self.client.chat.completions.create(
                model=self.model,
                response_model=LogicalValidationResult,
                messages=[
                    {"role": "system", "content": _SEMANTIC_SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.0,
                max_retries=self._max_retries,
            )
        except Exception as exc:  # noqa: BLE001 - errores de proveedor y de esquema
            raise RemoteValidationError(f"fallo en la validación semántica remota: {exc}") from exc

        if not result.is_valid and not result.rejection_reason.strip():
            result.rejection_reason = "motivo no especificado por el validador remoto"
        if not result.is_valid:
            result.rejection_reason = result.rejection_reason.strip()
        return result
