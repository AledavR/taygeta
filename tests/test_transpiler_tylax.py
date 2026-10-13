"""Pruebas del conversor TeX→Typst delegado a tylax.

Cubre el reemplazo del parser propio por la fachada :class:`MathConverter`
sobre ``tylax.latex_to_typst``: corpus de oro de fórmulas representativas,
elevación a :class:`TranspileError` del marcador de error sintáctico de tylax,
passthrough de comandos desconocidos y compilación real con el binario
``typst`` (se omite si no está en el PATH).
"""

from __future__ import annotations

import shutil

import pytest

from taygeta.core.transpiler import (
    MathConverter,
    TranspileError,
    compile_typst,
    markdown_to_typst,
    validate_typst,
)

#: Pares ``(TeX, salida Typst esperada)`` con las opciones fijas del módulo
#: (``frac_to_slash=False``, ``non_strict=True``).
_CORPUS: tuple[tuple[str, str], ...] = (
    (r"\frac{1}{2}", "frac(1, 2)"),
    (r"\frac{\partial f}{\partial x}", "frac(partial f, partial x)"),
    (r"\sqrt{x}", "sqrt(x)"),
    (r"\sqrt[3]{x}", "root(3, x)"),
    (r"\alpha\beta", "alpha beta"),
    (r"\Gamma(p+q)", "Gamma (p + q)"),
    (r"\mathbb{R}", "RR"),
    (r"\vec{v}", "arrow(v)"),
    (r"\bar{x}", "overline(x)"),
    (r"\sum_{i=1}^{n} i", "sum_(i = 1)^(n) i"),
    (r"\int_0^1 x\,dx", "integral_(0)^(1) x thin d x"),
    (r"\lim_{x\to 0} f(x)", "lim_(x -> 0) f (x)"),
    (r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}", 'mat(delim: "(", a, b ; c, d)'),
    (
        r"\begin{cases} x & x>0 \\ -x & x\le 0 \end{cases}",
        "cases(x & x > 0, - x & x <= 0)",
    ),
    (r"\begin{aligned} x &= 1 \\ y &= 2 \end{aligned}", "x & = 1 \\ y & = 2"),
    (r"\begin{matrix} 1 & 2 \\ 3 & 4 \end{matrix}", "mat(delim: #none, 1, 2 ; 3, 4)"),
    (r"\left| x \right|", "abs(x)"),
    (r"\left. \frac{1}{2} \right|_0^1", "lr(frac(1, 2) bar.v)_(0)^(1)"),
    (r"x \cdot y", "x dot y"),
    (r"\mid", "divides"),
    (r"\subseteq", "subset.eq"),
    (r"\notin", "in.not"),
    (r"\to", "->"),
    (r"x \text{ tal que } x>0", "x #text[tal que] x > 0"),
    (r"\mathrm{d}x", "dif x"),
    (r"\mathbf{A}", "upright(bold(A))"),
    (r"\hat{\theta}", "hat(theta)"),
    (r"\{ x \mid x>0 \}", "{ x divides x > 0 }"),
    (r"\emptyset", "emptyset"),
    (r"\varepsilon", "epsilon"),
    (r"\varphi", "phi"),
    (r"\ne", "!="),
    (r"\approx", "approx"),
)

_CORPUS_IDS = [tex for tex, _ in _CORPUS]


@pytest.mark.parametrize(("tex", "esperado"), _CORPUS, ids=_CORPUS_IDS)
def test_corpus_de_oro_tex_a_typst(tex: str, esperado: str) -> None:
    assert MathConverter().convert(tex) == esperado


@pytest.mark.skipif(shutil.which("typst") is None, reason="binario typst no disponible")
@pytest.mark.parametrize(("tex", "esperado"), _CORPUS, ids=_CORPUS_IDS)
def test_corpus_compila_con_typst_real(tex: str, esperado: str, tmp_path) -> None:
    result = compile_typst(
        f"$ {esperado} $\n", tmp_path, filename="main.typ", output_name="output.pdf"
    )
    assert result.ok, result.format_errors()


def test_latex_no_analizable_eleva_transpile_error() -> None:
    with pytest.raises(TranspileError) as exc_info:
        MathConverter().convert(r"}")
    assert exc_info.value.original == r"}"
    assert "tylax" in exc_info.value.message


def test_comando_desconocido_no_lanza() -> None:
    # tylax es no estricto: conserva el comando desconocido como texto crudo en
    # lugar de eliminarlo en silencio; el `typst compile` final lo detecta.
    salida = MathConverter().convert(r"\noop{foo}")
    assert "noop" in salida


def test_markdown_to_typst_integra_math_y_figuras() -> None:
    salida = markdown_to_typst(
        "Calcula $$\\int_0^1 x\\,dx$$ para $x > 0$.\n\n{{fig:1}}",
        figure_paths={"fig:1": "fig_1.png"},
    )
    assert "integral_(0)^(1)" in salida
    assert "$x > 0$" in salida
    assert '#figure(image("assets/fig_1.png"' in salida


@pytest.mark.skipif(shutil.which("typst") is None, reason="binario typst no disponible")
def test_validate_typst_acepta_documento_generado() -> None:
    fuente = markdown_to_typst("Resuelve $$\\frac{1}{2}$$ para $x>0$.")
    result = validate_typst(fuente)
    assert result.ok, result.format_errors()
