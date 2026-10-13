"""Pruebas de la sanitización defensiva de delimitadores matemáticos.

Cubre el fix de renderizado KaTeX/Typst: los modelos OCR/LLM pueden emitir
``\\[...\\]`` y ``\\(...\\)`` en lugar de ``$$...$$`` y ``$...$``. El esquema
canónico debe normalizarlos automáticamente al validar (ingesta y persistencia),
y los system prompts deben prohibir explícitamente los delimitadores no estándar.
"""

from __future__ import annotations

import pytest

from taygeta.agents.structurer import _REFINING_SYSTEM_PROMPT, _STRUCTURING_SYSTEM_PROMPT
from taygeta.core.schema import (
    EnrichedQuestion,
    QuestionRaw,
    deterministic_question_id,
    sanitize_math_delimiters,
    utc_now_iso,
)
from taygeta.ingest.ocr_client import OCR_SYSTEM_PROMPT


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (r"resuelve \[ x^2 = 4 \]", "resuelve $$ x^2 = 4 $$"),
        (r"sea \( x = 2 \)", "sea $ x = 2 $"),
        (r"\\[ a \\]", "$$ a $$"),
        (r"\( f(x) \\)", "$ f(x) $"),
        ("ya $ok$ y $$bloque$$", "ya $ok$ y $$bloque$$"),
        ("sin matematicas: [corchete]", "sin matematicas: [corchete]"),
        ("", ""),
    ],
)
def test_sanitize_math_delimiters(raw: str, expected: str) -> None:
    assert sanitize_math_delimiters(raw) == expected


def test_sanitize_is_idempotent() -> None:
    once = sanitize_math_delimiters(r"\[ a \] y \( b \)")
    assert once == "$$ a $$ y $ b $"
    assert sanitize_math_delimiters(once) == once


def test_question_raw_sanitizes_statement_and_choices() -> None:
    question = QuestionRaw(
        id=deterministic_question_id("doc.pdf", "0", "0"),
        statement_raw=r"Resuelve \[ x^2 = 4 \] para \( x > 0 \).",
        question_type="multiple_choice",
        choices=[
            {"id": "A", "content_raw": r"\( x = 2 \)"},
            {"id": "B", "content_raw": r"\[ x = -2 \]"},
        ],
    )
    assert question.statement_raw == "Resuelve $$ x^2 = 4 $$ para $ x > 0 $."
    assert question.choices is not None
    assert question.choices[0].content_raw == "$ x = 2 $"
    assert question.choices[1].content_raw == "$$ x = -2 $$"


def test_enriched_question_sanitizes_statement() -> None:
    question = EnrichedQuestion(
        id=deterministic_question_id("doc.pdf", "1", "0"),
        statement_raw=r"\[ \int_0^1 x\,dx \]",
        question_type="open_ended",
        course="calculo",
        topic="integrales",
        difficulty=3,
        created_at=utc_now_iso(),
    )
    assert question.statement_raw == r"$$ \int_0^1 x\,dx $$"


@pytest.mark.parametrize(
    "prompt",
    [_STRUCTURING_SYSTEM_PROMPT, _REFINING_SYSTEM_PROMPT, OCR_SYSTEM_PROMPT],
)
def test_prompts_enforce_standard_delimiters(prompt: str) -> None:
    assert "$...$" in prompt
    assert "$$...$$" in prompt
    assert "PROHIBIDO" in prompt
