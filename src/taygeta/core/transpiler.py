"""Transpilador Markdown/TeX a Typst y LaTeX (sección 10).

Diseño (sin sustituciones globales por regex):

* ``pandoc_to_ast`` invoca el CLI de Pandoc vía ``subprocess`` (argumentos en
  lista, nunca ``shell=True``) y devuelve el AST JSON del documento.
* ``_render_blocks`` / ``_render_inlines`` recorren recursivamente los nodos de
  bloque e inline conservando la estructura semántica.
* Los nodos ``Math`` (``InlineMath``/``DisplayMath``) se procesan con
  :class:`MathConverter`, un conversor TeX→Typst **sensible a tokens y llaves**
  (análisis sintáctico recursivo, no reemplazos por regex).
* Los placeholders ``{{fig:N}}`` se detectan con un escáner manual (no regex) y
  se convierten en bloques ``#figure(image("assets/...", width: 80%))``.
* La exportación a LaTeX usa Pandoc (``--to latex``) y ``\\includegraphics``
  para las figuras.
* ``compile_typst`` / ``validate_typst`` ejecutan el binario ``typst`` en modo
  real de compilación y devuelven los errores con contexto.

Los constructos no soportados producen :class:`TranspileError` descriptivo y
conservan el TeX original para recuperación o exportación LaTeX.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from taygeta.core.schema import sanitize_math_delimiters, sanitize_norm_bars

__all__ = [
    "CompileResult",
    "MathConverter",
    "TranspileError",
    "compile_typst",
    "markdown_to_latex",
    "markdown_to_typst",
    "pandoc_to_ast",
    "sanitize_math_for_typst",
    "validate_typst",
]


class TranspileError(RuntimeError):
    """Error controlado de conversión, conservando el TeX original cuando aplica."""

    def __init__(self, message: str, *, original: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.original = original


@dataclass(slots=True)
class CompileResult:
    """Resultado de una compilación real con Typst."""

    ok: bool
    pdf_path: Path | None
    stdout: str
    stderr: str
    returncode: int

    def format_errors(self) -> str:
        """Devuelve el mensaje de error de Typst con contexto, si lo hubo."""
        return self.stderr.strip() or "sin errores"


# ---------------------------------------------------------------------------
# Pandoc
# ---------------------------------------------------------------------------


def pandoc_to_ast(markdown: str, *, pandoc_bin: str = "pandoc") -> dict[str, Any]:
    """Obtiene el AST JSON de Pandoc para un documento Markdown.

    Raises:
        TranspileError: si Pandoc no está disponible, falla o devuelve JSON
            inválido.
    """
    executable = shutil.which(pandoc_bin) or pandoc_bin
    process = subprocess.run(
        [executable, "--from=markdown", "--to=json"],
        input=markdown,
        text=True,
        capture_output=True,
    )
    if process.returncode != 0:
        raise TranspileError(f"pandoc falló: {process.stderr.strip()}")
    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise TranspileError(f"pandoc devolvió JSON inválido: {exc}") from exc


# ---------------------------------------------------------------------------
# Conversor TeX → Typst (sensible a tokens y llaves)
# ---------------------------------------------------------------------------


#: Comandos de espaciado fino que no alteran la semántica (se descartan).
_SPACING = frozenset({",", ";", ":", "!", " ", "~"})

#: Comandos transparentes de tamaño de delimitador (Typst los auto-dimensiona).
_TRANSPARENT = frozenset(
    {"left", "right", "middle", "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr"}
)

#: Caracteres escapados con barra invertida.
_ESCAPED = {"%": "%", "$": "$", "#": "#", "&": "&", "{": "{", "}": "}"}

#: Letras griegas y sus nombres en Typst.
_GREEK = {
    "alpha": "alpha",
    "beta": "beta",
    "gamma": "gamma",
    "delta": "delta",
    "epsilon": "epsilon",
    "varepsilon": "epsilon",
    "zeta": "zeta",
    "eta": "eta",
    "theta": "theta",
    "vartheta": "theta",
    "iota": "iota",
    "kappa": "kappa",
    "lambda": "lambda",
    "mu": "mu",
    "nu": "nu",
    "xi": "xi",
    "omicron": "omicron",
    "pi": "pi",
    "varpi": "pi",
    "rho": "rho",
    "varrho": "rho",
    "sigma": "sigma",
    "varsigma": "sigma",
    "tau": "tau",
    "upsilon": "upsilon",
    "phi": "phi",
    "varphi": "phi",
    "chi": "chi",
    "psi": "psi",
    "omega": "omega",
    "Gamma": "Gamma",
    "Delta": "Delta",
    "Theta": "Theta",
    "Lambda": "Lambda",
    "Xi": "Xi",
    "Pi": "Pi",
    "Sigma": "Sigma",
    "Upsilon": "Upsilon",
    "Phi": "Phi",
    "Psi": "Psi",
    "Omega": "Omega",
}

#: Símbolos y relaciones usuales.
_SYMBOLS = {
    "infty": "infinity",
    "to": "->",
    "rightarrow": "->",
    "leftarrow": "<-",
    "gets": "<-",
    "cdot": "dot.op",
    "times": "times",
    "ast": "ast",
    "pm": "plus.minus",
    "mp": "minus.plus",
    "le": "<=",
    "leq": "<=",
    "ge": ">=",
    "geq": ">=",
    "ne": "!=",
    "neq": "!=",
    "approx": "approx",
    "equiv": "equiv",
    "sim": "tilde.op",
    "cong": "tilde.equiv",
    "propto": "prop",
    "partial": "diff",
    "nabla": "nabla",
    "in": "in",
    "ni": "in",
    "notin": "not in",
    "cup": "union",
    "cap": "inter",
    "subseteq": "subset.eq",
    "subset": "subset",
    "supset": "supset",
    "supseteq": "supset.eq",
    "emptyset": "emptyset",
    "varnothing": "emptyset",
    "forall": "forall",
    "exists": "exists",
    "neg": "not",
    "land": "and",
    "wedge": "and",
    "lor": "or",
    "vee": "or",
    "implies": "==>",
    "iff": "<=>",
    "Rightarrow": "=>",
    "Leftrightarrow": "<=>",
    "mid": "|",
    "colon": ":",
}

#: Operadores y funciones que no consumen argumentos (sus scripts se adjuntan).
_OPERATORS = {
    "sum": "sum",
    "prod": "prod",
    "int": "integral",
    "iint": "integral.double",
    "iiint": "integral.triple",
    "oint": "integral.cont",
    "lim": "lim",
    "limsup": "limsup",
    "liminf": "liminf",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "cot": "cot",
    "sec": "sec",
    "csc": "csc",
    "arcsin": "arcsin",
    "arccos": "arccos",
    "arctan": "arctan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "log": "log",
    "ln": "ln",
    "lg": "lg",
    "exp": "exp",
    "min": "min",
    "max": "max",
    "det": "det",
    "gcd": "gcd",
    "deg": "deg",
    "sup": "sup",
    "inf": "inf",
    "arg": "arg",
    "dim": "dim",
    "ker": "ker",
    "Pr": "Pr",
}

#: Comandos de estilo/ornamento que aplican a un único grupo.
_STYLE = {
    "mathrm": "upright",
    "mathbf": "bold",
    "boldsymbol": "bold",
    "mathit": "italic",
    "mathcal": "cal",
    "mathscr": "cal",
    "mathbb": "bb",
    "mathsf": "sans",
    "mathtt": "mono",
    "hat": "hat",
    "widehat": "hat",
    "bar": "bar",
    "overline": "overline",
    "underline": "underline",
    "vec": "arrow",
    "tilde": "tilde",
    "widetilde": "tilde",
    "dot": "dot",
    "ddot": "ddot",
}

#: Entornos matriciales soportados y su delimitador Typst (``None`` = paréntesis).
_MATRIX_ENVS = {
    "pmatrix": None,
    "bmatrix": '"["',
    "Bmatrix": '"{"',
    "vmatrix": '"|"',
    "Vmatrix": '"||"',
    "matrix": "#none",
    "smallmatrix": "#none",
    "array": "#none",
}

_FRACTIONS = frozenset({"frac", "dfrac", "tfrac", "cfrac"})

#: Comandos de "modo texto" dentro de matemáticas. Su argumento es texto literal
#: (con espacios) y debe convertirse en una cadena Typst ``"..."``, nunca en un
#: identificador: ``\text{ si }`` → ``upright(" si ")``, jamás ``upright(si)``.
_TEXT_COMMANDS = frozenset({"text", "textrm", "textnormal", "textup", "mbox"})


def _quote_typst_string(text: str) -> str:
    """Entrecomilla ``text`` como literal de cadena Typst, escapando ``"`` y ``\\``.

    El contenido de ``\text{...}`` es texto literal; Typst lo interpreta como
    cadena (``upright(" si ")``) en lugar de como variables (``upright(si)``).
    """
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _scan_text_argument(tex: str, index: int) -> tuple[tuple[str, str], int] | None:
    """Captura el argumento literal ``{...}`` de un comando de modo texto.

    Devuelve el token ``("text", contenido_crudo)`` y el índice tras la llave de
    cierre, conservando los espacios interiores intactos. Si el comando no va
    seguido de un grupo ``{...}`` bien formado devuelve ``None``, en cuyo caso se
    tratará como un comando normal.
    """
    length = len(tex)
    cursor = index
    while cursor < length and tex[cursor] in " \t\r\n":
        cursor += 1
    if cursor >= length or tex[cursor] != "{":
        return None
    depth = 1
    start = cursor + 1
    scan = start
    while scan < length and depth > 0:
        if tex[scan] == "{":
            depth += 1
        elif tex[scan] == "}":
            depth -= 1
            if depth == 0:
                break
        scan += 1
    if depth != 0:
        return None
    return ("text", tex[start:scan]), scan + 1


def _tokenize_tex(tex: str) -> list[tuple[str, str]]:
    """Convierte TeX matemático en una lista de tokens ``(tipo, valor)``."""
    tokens: list[tuple[str, str]] = []
    index = 0
    length = len(tex)
    while index < length:
        char = tex[index]
        if char in " \t\r\n":
            index += 1
            continue
        if char == "\\":
            if index + 1 < length and tex[index + 1] == "\\":
                tokens.append(("newline", "\\\\"))
                index += 2
                continue
            if index + 1 < length:
                following = tex[index + 1]
                if following.isalpha():
                    end = index + 1
                    while end < length and tex[end].isalpha():
                        end += 1
                    name = tex[index + 1 : end]
                    if name in _TEXT_COMMANDS:
                        text_argument = _scan_text_argument(tex, end)
                        if text_argument is not None:
                            text_token, index = text_argument
                            tokens.append(text_token)
                            continue
                    tokens.append(("cmd", name))
                    index = end
                else:
                    tokens.append(("cmd", following))
                    index += 2
                continue
            tokens.append(("cmd", " "))
            index += 1
            continue
        if char == "{":
            tokens.append(("open", "{"))
        elif char == "}":
            tokens.append(("close", "}"))
        elif char == "&":
            tokens.append(("align", "&"))
        elif char in "^_":
            tokens.append(("script", char))
        else:
            tokens.append(("char", char))
        index += 1
    return tokens


class MathConverter:
    """Conversor estructural de una fórmula TeX a Typst."""

    def __init__(self) -> None:
        self._original = ""

    def convert(self, tex: str) -> str:
        """Convierte una fórmula TeX (sin delimitadores ``$``) a Typst."""
        self._original = tex
        tokens = _tokenize_tex(tex)
        result, index = self._parse_sequence(tokens, 0, frozenset())
        if index < len(tokens):
            raise TranspileError(
                f"token no esperado en la fórmula: {tokens[index]!r}", original=tex
            )
        return result

    # -- Análisis sintáctico ------------------------------------------------
    @staticmethod
    def _is_stop(token: tuple[str, str], stop: frozenset[str]) -> bool:
        kind, value = token
        if kind in stop:
            return True
        return kind == "cmd" and value == "end" and "end" in stop

    @staticmethod
    def _needs_space(previous: str, following: str) -> bool:
        """Indica si dos átomos alfanuméricos adyacentes requieren un separador.

        Typst une letras adyacentes en un único identificador, así que
        ``\\alpha\\beta`` debe convertirse en ``alpha beta`` y ``x\\,dx`` en
        ``x d x``. Un operador (``+``, ``=``, ``->``...) ya actúa de separador.
        """
        return (
            bool(previous) and bool(following) and previous[-1].isalnum() and following[0].isalnum()
        )

    def _parse_sequence(
        self,
        tokens: list[tuple[str, str]],
        index: int,
        stop: frozenset[str],
        *,
        sep: bool = True,
    ):
        parts: list[str] = []
        while index < len(tokens):
            token = tokens[index]
            if self._is_stop(token, stop):
                break
            kind, value = token
            if kind == "close":
                raise TranspileError("llave '}' sin abrir", original=self._original)
            if kind == "align":
                raise TranspileError(
                    "alineación '&' fuera de una matriz o casos", original=self._original
                )
            if kind == "newline":
                raise TranspileError(
                    "salto de línea '\\\\' fuera de una matriz o casos", original=self._original
                )
            if kind == "script":
                raise TranspileError(
                    "superíndice/subíndice sin base previa", original=self._original
                )
            atom, index = self._parse_atom(tokens, index)
            while index < len(tokens) and tokens[index][0] == "script":
                operator = tokens[index][1]
                index += 1
                argument, index = self._parse_script_argument(tokens, index)
                atom += "^" + argument if operator == "^" else "_" + argument
            if sep and parts and self._needs_space(parts[-1], atom):
                parts.append(" ")
            parts.append(atom)
        return "".join(parts), index

    def _parse_atom(self, tokens: list[tuple[str, str]], index: int):
        if index >= len(tokens):
            raise TranspileError("fórmula truncada", original=self._original)
        kind, value = tokens[index]
        if kind == "open":
            return self._parse_group(tokens, index)
        if kind == "cmd":
            return self._parse_command(tokens, index)
        if kind == "char":
            return value, index + 1
        if kind == "text":
            return f"upright({_quote_typst_string(value)})", index + 1
        raise TranspileError(f"token no esperado {tokens[index]!r}", original=self._original)

    def _parse_group(self, tokens: list[tuple[str, str]], index: int, *, sep: bool = True):
        # tokens[index] == ("open", "{")
        index += 1
        content, index = self._parse_sequence(tokens, index, frozenset({"close"}), sep=sep)
        if index < len(tokens) and tokens[index][0] == "close":
            index += 1
        else:
            raise TranspileError("llave '{' sin cerrar", original=self._original)
        return content, index

    def _require_group(self, tokens: list[tuple[str, str]], index: int, *, sep: bool = True):
        if index >= len(tokens) or tokens[index][0] != "open":
            raise TranspileError("se esperaba un grupo '{...}'", original=self._original)
        return self._parse_group(tokens, index, sep=sep)

    def _parse_script_argument(self, tokens: list[tuple[str, str]], index: int):
        if index >= len(tokens):
            raise TranspileError("script sin argumento", original=self._original)
        if tokens[index][0] == "open":
            content, index = self._parse_group(tokens, index)
            return f"({content})", index
        atom, index = self._parse_atom(tokens, index)
        return f"({atom})", index

    def _parse_optional_bracket(self, tokens: list[tuple[str, str]], index: int):
        # tokens[index] == ("char", "[")
        index += 1
        chars: list[str] = []
        while index < len(tokens) and not (tokens[index][0] == "char" and tokens[index][1] == "]"):
            chars.append(tokens[index][1])
            index += 1
        if index < len(tokens) and tokens[index][0] == "char" and tokens[index][1] == "]":
            index += 1
        else:
            raise TranspileError("corchete ']' sin cerrar", original=self._original)
        return "".join(chars), index

    def _parse_substack(self, tokens: list[tuple[str, str]], index: int):
        # tokens[index] == ("open", "{")
        index += 1
        items: list[str] = []
        stop = frozenset({"newline", "close"})
        while index < len(tokens):
            token = tokens[index]
            if token[0] == "close":
                break
            cell, index = self._parse_sequence(tokens, index, stop)
            items.append(cell)
            if index < len(tokens) and tokens[index][0] == "newline":
                index += 1
                continue
        if index < len(tokens) and tokens[index][0] == "close":
            index += 1
        else:
            raise TranspileError("\\substack sin grupo de cierre", original=self._original)
        return items, index

    def _parse_environment(self, tokens: list[tuple[str, str]], index: int):
        # tokens[index] == ("open", "{") tras el comando \begin
        if index >= len(tokens) or tokens[index][0] != "open":
            raise TranspileError("\\begin sin nombre de entorno", original=self._original)
        index += 1
        name_chars: list[str] = []
        while index < len(tokens) and tokens[index][0] == "char":
            name_chars.append(tokens[index][1])
            index += 1
        if index < len(tokens) and tokens[index][0] == "close":
            index += 1
        else:
            raise TranspileError("\\begin sin llave de cierre", original=self._original)
        environment = "".join(name_chars)

        rows: list[list[str]] = []
        current: list[str] = []
        stop = frozenset({"align", "newline", "end"})
        while index < len(tokens):
            token = tokens[index]
            if token[0] == "cmd" and token[1] == "end":
                break
            cell, index = self._parse_sequence(tokens, index, stop)
            current.append(cell)
            if index >= len(tokens):
                break
            token = tokens[index]
            if token[0] == "align":
                index += 1
                continue
            if token[0] == "newline":
                index += 1
                rows.append(current)
                current = []
                continue
            break
        if current:
            rows.append(current)

        if index < len(tokens) and tokens[index][0] == "cmd" and tokens[index][1] == "end":
            index += 1
            if index < len(tokens) and tokens[index][0] == "open":
                index += 1
                while index < len(tokens) and tokens[index][0] == "char":
                    index += 1
                if index < len(tokens) and tokens[index][0] == "close":
                    index += 1
        else:
            raise TranspileError(f"entorno '{environment}' sin \\end", original=self._original)
        return self._render_environment(environment, rows), index

    def _render_environment(self, environment: str, rows: list[list[str]]) -> str:
        if environment == "cases":
            cells = [cell for row in rows for cell in row]
            return "cases(" + ", ".join(cells) + ")"
        if environment == "aligned":
            # `x &= 1` alinea en el `=`: se concatenan las celdas de cada fila.
            body = "; ".join("".join(row) for row in rows)
            return "mat(" + body + ")"
        if environment in _MATRIX_ENVS:
            body = "; ".join(", ".join(row) for row in rows)
            delimiter = _MATRIX_ENVS[environment]
            if delimiter is None:
                return "mat(" + body + ")"
            return f"mat(delim: {delimiter}, " + body + ")"
        raise TranspileError(f"entorno TeX no soportado: {environment}", original=self._original)

    # -- Despacho de comandos ----------------------------------------------
    def _parse_command(self, tokens: list[tuple[str, str]], index: int):
        _, name = tokens[index]
        index += 1

        if name in _SPACING:
            return "" if name == "!" else " ", index
        if name in ("quad", "qquad"):
            return " ", index
        if name in _TRANSPARENT:
            return "", index
        if name in _ESCAPED:
            return _ESCAPED[name], index
        if name in _STYLE:
            argument, index = self._require_group(tokens, index, sep=False)
            return f"{_STYLE[name]}({argument})", index
        if name in _GREEK:
            return _GREEK[name], index
        if name in _SYMBOLS:
            return _SYMBOLS[name], index
        if name in _OPERATORS:
            return _OPERATORS[name], index
        if name in _FRACTIONS:
            numerator, index = self._require_group(tokens, index)
            denominator, index = self._require_group(tokens, index)
            return f"frac({numerator}, {denominator})", index
        if name == "sqrt":
            degree: str | None = None
            if index < len(tokens) and tokens[index][0] == "char" and tokens[index][1] == "[":
                degree, index = self._parse_optional_bracket(tokens, index)
            argument, index = self._require_group(tokens, index)
            if degree is not None:
                return f"root({degree}, {argument})", index
            return f"sqrt({argument})", index
        if name == "substack":
            items, index = self._parse_substack(tokens, index)
            return "std.stack(" + ", ".join(items) + ")", index
        if name == "begin":
            return self._parse_environment(tokens, index)
        if name == "end":
            raise TranspileError("\\end sin \\begin correspondiente", original=self._original)
        raise TranspileError(f"comando TeX no soportado: \\{name}", original=self._original)


# ---------------------------------------------------------------------------
# Placeholders de figura (escáner manual, sin regex)
# ---------------------------------------------------------------------------


def _scan_figure_placeholders(text: str) -> list[tuple[str, str]]:
    """Devuelve una lista de pares ``(tipo, valor)``: ``("text", ...)`` o ``("figure", n)``."""
    spans: list[tuple[str, str]] = []
    marker = "{{fig:"
    index = 0
    while index < len(text):
        found = text.find(marker, index)
        if found == -1:
            spans.append(("text", text[index:]))
            break
        if found > index:
            spans.append(("text", text[index:found]))
        cursor = found + len(marker)
        digits = ""
        while cursor < len(text) and text[cursor].isdigit():
            digits += text[cursor]
            cursor += 1
        if digits and cursor + 1 < len(text) and text[cursor : cursor + 2] == "}}":
            spans.append(("figure", digits))
            index = cursor + 2
        else:
            spans.append(("text", text[index : index + 1]))
            index = index + 1
    return spans


def _figure_asset(number: str, figure_paths: Mapping[str, str]) -> str:
    key = f"fig:{number}"
    return figure_paths.get(key, f"fig_{number}.png")


def _figure_block(number: str, figure_paths: Mapping[str, str]) -> str:
    asset = _figure_asset(number, figure_paths)
    return f'#figure(image("assets/{asset}", width: 80%))'


# ---------------------------------------------------------------------------
# Renderizado del AST de Pandoc
# ---------------------------------------------------------------------------


def _render_str(text: str, figure_paths: Mapping[str, str]) -> str:
    parts: list[str] = []
    for kind, value in _scan_figure_placeholders(text):
        if kind == "figure":
            parts.append("\n\n" + _figure_block(value, figure_paths) + "\n\n")
        else:
            parts.append(value)
    return "".join(parts)


def _render_inlines(
    inlines: list[dict[str, Any]], converter: MathConverter, figure_paths: Mapping[str, str]
) -> str:
    parts: list[str] = []
    for node in inlines:
        node_type = node.get("t")
        content = node.get("c")
        if node_type == "Str":
            parts.append(_render_str(content, figure_paths))
        elif node_type == "Space":
            parts.append(" ")
        elif node_type == "SoftBreak":
            parts.append(" ")
        elif node_type == "LineBreak":
            parts.append("\n")
        elif node_type == "Emph":
            parts.append("_" + _render_inlines(content, converter, figure_paths) + "_")
        elif node_type == "Strong":
            parts.append("*" + _render_inlines(content, converter, figure_paths) + "*")
        elif node_type == "Code":
            parts.append("`" + content[1] + "`")
        elif node_type == "Link":
            text = _render_inlines(content[1], converter, figure_paths)
            parts.append(f'#link("{content[2][0]}")[{text}]')
        elif node_type == "Math":
            parts.append(_render_math(content, converter))
        elif node_type == "RawInline":
            parts.append(_render_raw_inline(content, converter))
        elif node_type == "Quoted":
            parts.append('"' + _render_inlines(content[1], converter, figure_paths) + '"')
        elif node_type == "Span":
            parts.append(_render_inlines(content[1], converter, figure_paths))
        elif node_type == "Note":
            continue
        else:
            raise TranspileError(f"elemento inline no soportado: {node_type}")
    return "".join(parts)


def _render_math(content: list[Any], converter: MathConverter) -> str:
    math_type = content[0].get("t")
    tex = content[1]
    typst = converter.convert(tex)
    if math_type == "DisplayMath":
        return "\n$ " + typst + " $\n"
    return "$" + typst + "$"


def _render_raw_inline(content: list[Any], converter: MathConverter) -> str:
    fmt = content[0]
    text = content[1]
    if fmt in ("tex", "latex"):
        return "$" + converter.convert(text) + "$"
    return text


def _render_list_item(
    item: list[dict[str, Any]], converter: MathConverter, figure_paths: Mapping[str, str]
) -> str:
    return _render_blocks(item, converter, figure_paths).strip()


def _render_block(
    block: dict[str, Any], converter: MathConverter, figure_paths: Mapping[str, str]
) -> str:
    node_type = block.get("t")
    content = block.get("c")
    if node_type in ("Para", "Plain"):
        return _render_inlines(content, converter, figure_paths)
    if node_type == "Header":
        level = content[0]
        text = _render_inlines(content[2], converter, figure_paths)
        return "=" * level + " " + text
    if node_type == "BulletList":
        items = ["- " + _render_list_item(item, converter, figure_paths) for item in content]
        return "\n".join(items)
    if node_type == "OrderedList":
        rendered = ["+ " + _render_list_item(item, converter, figure_paths) for item in content[1]]
        return "\n".join(rendered)
    if node_type == "CodeBlock":
        return "```\n" + content[1] + "\n```"
    if node_type == "BlockQuote":
        inner = _render_blocks(content, converter, figure_paths)
        return "\n".join("> " + line for line in inner.splitlines())
    if node_type == "HorizontalRule":
        return ""
    if node_type == "Div":
        return _render_blocks(content[1], converter, figure_paths)
    if node_type == "Null":
        return ""
    raise TranspileError(f"bloque no soportado: {node_type}")


def _render_blocks(
    blocks: list[dict[str, Any]], converter: MathConverter, figure_paths: Mapping[str, str]
) -> str:
    rendered = [_render_block(block, converter, figure_paths) for block in blocks]
    return "\n\n".join(part for part in rendered if part)


# ---------------------------------------------------------------------------
# Sanitización de matemáticas para Typst
# ---------------------------------------------------------------------------


def _replace_empty_identifier(tex: str) -> str:
    """Reescribe la palabra ``empty`` como ``\\emptyset`` dentro de una fórmula.

    Algunos modelos OCR/LLM escriben ``empty`` (nombre del conjunto vacío en otras
    sintaxis) donde Typst espera ``emptyset``; el transpilador no reconoce ese
    identificador y Typst aborta con ``unknown variable: empty``. Solo se
    reemplaza la palabra completa, de modo que ``\\emptyset`` (cuyo identificador
    es ``emptyset``, no ``empty``) queda intacto. También se normaliza el comando
    inválido ``\\empty`` a ``\\emptyset``.
    """
    parts: list[str] = []
    index = 0
    length = len(tex)
    while index < length:
        char = tex[index]
        if char.isalpha():
            end = index
            while end < length and tex[end].isalnum():
                end += 1
            word = tex[index:end]
            parts.append(r"\emptyset" if word == "empty" else word)
            index = end
            continue
        if char == "\\" and index + 1 < length and tex[index + 1].isalpha():
            end = index + 1
            while end < length and tex[end].isalpha():
                end += 1
            command = tex[index + 1 : end]
            parts.append(r"\emptyset" if command == "empty" else tex[index:end])
            index = end
            continue
        parts.append(char)
        index += 1
    return "".join(parts)


def _find_inline_math_close(text: str, start: int) -> int:
    """Devuelve el índice del ``$`` que cierra una fórmula inline (o ``-1``).

    Se ignoran los dólares escapados (``\\$``) que pudieran aparecer dentro de la
    fórmula.
    """
    index = start
    length = len(text)
    while index < length:
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == "$":
            return index
        index += 1
    return -1


def _rewrite_empty_set(markdown: str) -> str:
    """Reemplaza ``empty`` por ``\\emptyset`` dentro de los bloques matemáticos.

    Recorre el Markdown distinguiendo display (``$$...$$``) e inline (``$...$``) y
    aplica :func:`_replace_empty_identifier` solo al interior de las fórmulas, de
    modo que la palabra inglesa ``empty`` del texto normal nunca se altera.
    """
    parts: list[str] = []
    index = 0
    length = len(markdown)
    while index < length:
        char = markdown[index]
        if char == "\\" and index + 1 < length and markdown[index + 1] == "$":
            parts.append(markdown[index : index + 2])
            index += 2
            continue
        if char != "$":
            parts.append(char)
            index += 1
            continue
        if markdown.startswith("$$", index):
            close = markdown.find("$$", index + 2)
            if close == -1:
                parts.append(markdown[index:])
                break
            parts.append("$$")
            parts.append(_replace_empty_identifier(markdown[index + 2 : close]))
            parts.append("$$")
            index = close + 2
            continue
        close = _find_inline_math_close(markdown, index + 1)
        if close == -1:
            parts.append(markdown[index:])
            break
        parts.append("$")
        parts.append(_replace_empty_identifier(markdown[index + 1 : close]))
        parts.append("$")
        index = close + 1
    return "".join(parts)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def sanitize_math_for_typst(markdown: str) -> str:
    """Sanitiza el Markdown de entrada justo antes de compilar a Typst.

    Endurecimiento defensivo del compilador: aunque los esquemas ya normalizan
    los delimitadores al validar (y la base de datos almacena la forma canónica),
    este paso garantiza que cualquier enunciado que llegue al transpilador quede
    listo para Typst:

    * ``\\[...\\]`` y ``\\(...\\)`` se reescriben a ``$$...$$`` / ``$...$`` (con
      :func:`~taygeta.core.schema.sanitize_math_delimiters`), de modo que Pandoc
      genere nodos ``DisplayMath``/``InlineMath`` y el renderizado produzca los
      delimitadores de bloque ``$ ... $`` de Typst;
    * ``\\|`` (barra de norma LaTeX) se reescribe como ``||`` (norma Typst) con
      :func:`~taygeta.core.schema.sanitize_norm_bars`;
    * la palabra ``empty`` dentro de fórmulas se reescribe como ``\\emptyset``
      (símbolo del conjunto vacío) para evitar el error de Typst
      ``unknown variable: empty``.
    """
    sanitized = sanitize_norm_bars(sanitize_math_delimiters(markdown))
    return _rewrite_empty_set(sanitized)


def markdown_to_typst(
    markdown: str,
    *,
    figure_paths: Mapping[str, str] | None = None,
    pandoc_bin: str = "pandoc",
) -> str:
    """Convierte Markdown (con TeX ``$...$``/``$$...$$``) a Typst.

    Args:
        markdown: Documento Markdown con matemáticas TeX y placeholders ``{{fig:N}}``.
        figure_paths: Mapeo opcional ``fig:id`` → nombre de asset relativo a ``assets/``.
        pandoc_bin: Ruta del ejecutable de Pandoc.

    Raises:
        TranspileError: si hay un bloque/inline/constructo no soportado.
    """
    sanitized = sanitize_math_for_typst(markdown)
    ast = pandoc_to_ast(sanitized, pandoc_bin=pandoc_bin)
    converter = MathConverter()
    return _render_blocks(ast.get("blocks", []), converter, figure_paths or {}) + "\n"


_LATEX_FIG_SENTINEL = "TAYGETAFIG"


def _protect_figures(markdown: str) -> tuple[str, dict[str, str]]:
    """Sustituye ``{{fig:N}}`` por sentinel alfanuméricos antes de Pandoc.

    Pandoc escaparía las llaves en la salida LaTeX (``\\{\\{fig:1\\}\\}``), así que
    los placeholders se reemplazan por tokens alfanuméricos que Pandoc respeta.
    """
    parts: list[str] = []
    mapping: dict[str, str] = {}
    for kind, value in _scan_figure_placeholders(markdown):
        if kind == "figure":
            sentinel = _LATEX_FIG_SENTINEL + value
            mapping[sentinel] = value
            parts.append(sentinel)
        else:
            parts.append(value)
    return "".join(parts), mapping


def _replace_latex_sentinels(
    text: str, mapping: dict[str, str], figure_paths: Mapping[str, str]
) -> str:
    parts: list[str] = []
    index = 0
    while index < len(text):
        found = text.find(_LATEX_FIG_SENTINEL, index)
        if found == -1:
            parts.append(text[index:])
            break
        parts.append(text[index:found])
        cursor = found + len(_LATEX_FIG_SENTINEL)
        digits = ""
        while cursor < len(text) and text[cursor].isdigit():
            digits += text[cursor]
            cursor += 1
        sentinel = _LATEX_FIG_SENTINEL + digits
        if sentinel in mapping:
            asset = _figure_asset(mapping[sentinel], figure_paths)
            parts.append(
                "\\begin{center}\n\\includegraphics[width=0.8\\linewidth]"
                f"{{assets/{asset}}}\n\\end{{center}}"
            )
            index = cursor
        else:
            parts.append(text[index : index + 1])
            index += 1
    return "".join(parts)


def markdown_to_latex(
    markdown: str,
    *,
    figure_paths: Mapping[str, str] | None = None,
    pandoc_bin: str = "pandoc",
) -> str:
    """Convierte Markdown a LaTeX usando Pandoc y ``\\includegraphics`` para figuras."""
    protected, mapping = _protect_figures(markdown)
    executable = shutil.which(pandoc_bin) or pandoc_bin
    process = subprocess.run(
        [executable, "--from=markdown", "--to=latex"],
        input=protected,
        text=True,
        capture_output=True,
    )
    if process.returncode != 0:
        raise TranspileError(f"pandoc falló al exportar LaTeX: {process.stderr.strip()}")
    return _replace_latex_sentinels(process.stdout, mapping, figure_paths or {})


def compile_typst(
    source: str,
    workdir: str | Path,
    *,
    typst_bin: str = "typst",
    filename: str = "main.typ",
    output_name: str = "output.pdf",
) -> CompileResult:
    """Compila una fuente Typst a PDF en ``workdir`` usando el binario real.

    No usa ``shell=True``: invoca ``typst compile`` con argumentos en lista.
    """
    directory = Path(workdir)
    directory.mkdir(parents=True, exist_ok=True)
    main_file = directory / filename
    main_file.write_text(source)
    executable = shutil.which(typst_bin) or typst_bin
    output = directory / output_name
    process = subprocess.run(
        [executable, "compile", str(main_file), str(output)],
        text=True,
        capture_output=True,
        cwd=str(directory),
    )
    return CompileResult(
        ok=process.returncode == 0,
        pdf_path=output if process.returncode == 0 else None,
        stdout=process.stdout,
        stderr=process.stderr,
        returncode=process.returncode,
    )


def validate_typst(
    source: str,
    *,
    typst_bin: str = "typst",
    filename: str = "main.typ",
) -> CompileResult:
    """Valida una fuente Typst compilándola en un directorio temporal."""
    with tempfile.TemporaryDirectory(prefix="taygeta-typst-") as directory:
        return compile_typst(
            source, directory, typst_bin=typst_bin, filename=filename, output_name="output.pdf"
        )
