"""Transpilador Markdown/TeX a Typst y LaTeX (sección 10).

Diseño (sin sustituciones globales por regex):

* ``pandoc_to_ast`` invoca el CLI de Pandoc vía ``subprocess`` (argumentos en
  lista, nunca ``shell=True``) y devuelve el AST JSON del documento.
* ``_render_blocks`` / ``_render_inlines`` recorren recursivamente los nodos de
  bloque e inline conservando la estructura semántica.
* Los nodos ``Math`` (``InlineMath``/``DisplayMath``) se procesan con
  :class:`MathConverter`, una fachada sobre **tylax** (conversor LaTeX↔Typst
  escrito en Rust): cada fórmula TeX se delega a ``tylax.latex_to_typst`` con
  ``frac_to_slash=False`` y ``non_strict=True``.
* Los placeholders ``{{fig:N}}`` se detectan con un escáner manual (no regex) y
  se convierten en bloques ``#figure(image("assets/...", width: 80%))``.
* La exportación a LaTeX usa Pandoc (``--to latex``) y ``\\includegraphics``
  para las figuras.
* ``compile_typst`` / ``validate_typst`` ejecutan el binario ``typst`` en modo
  real de compilación y devuelven los errores con contexto.

Nota sobre el manejo de errores: tylax **no lanza** ante comandos TeX
desconocidos; con ``non_strict=True`` los conserva como texto crudo (en modo
estricto los elimina en silencio), de modo que un comando desconocido se
detecta en la compilación real con Typst (``unknown variable: ...``) y un
examen roto nunca genera un PDF sin aviso. Un LaTeX que tylax no puede analizar
produce el marcador ``/* LaTeX Error ... */``, que
:meth:`MathConverter.convert` eleva a :class:`TranspileError` conservando el
TeX original para recuperación o exportación LaTeX.
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

import tylax

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
# Conversor TeX → Typst (delegado a tylax)
# ---------------------------------------------------------------------------


#: Opciones fijas de conversión para las fórmulas del banco.
#:
#: * ``frac_to_slash=False`` mantiene ``frac(a, b)`` (estilo del conversor
#:   anterior) en lugar del atajo ``a/b``.
#: * ``non_strict=True`` (valor por defecto de tylax) conserva los comandos
#:   desconocidos como texto crudo en vez de eliminarlos en silencio: así el
#:   ``typst compile`` final los detecta y un enunciado roto nunca genera un
#:   PDF sin aviso (invariante del dominio).
_L2T_OPTIONS = tylax.L2TOptions(frac_to_slash=False, non_strict=True)

#: Marcador que tylax inserta cuando la entrada LaTeX no se pudo analizar.
_LATEX_ERROR_MARK = "/* LaTeX Error"


class MathConverter:
    """Conversor estructural de una fórmula TeX a Typst (fachada sobre tylax).

    tylax (conversor LaTeX↔Typst escrito en Rust) traduce cada fórmula con un
    catálogo de símbolos y expansión de macros mucho más amplios que el parser
    propio que reemplaza. La clase conserva la API pública anterior
    (``convert(tex)``) y el atributo ``_original`` para que los renderizadores
    de bloques/inlines del módulo no cambien.
    """

    def __init__(self) -> None:
        self._original = ""

    def convert(self, tex: str) -> str:
        """Convierte una fórmula TeX (sin delimitadores ``$``) a Typst.

        Raises:
            TranspileError: si tylax no pudo analizar el LaTeX de entrada (la
                salida contiene el marcador ``/* LaTeX Error ... */``). Los
                comandos desconocidos **no** lanzan: se conservan como texto
                crudo y la compilación real con Typst los detectará.
        """
        self._original = tex
        result = tylax.latex_to_typst(tex, options=_L2T_OPTIONS)
        if _LATEX_ERROR_MARK in result:
            raise TranspileError(f"LaTeX no analizable por tylax: {result.strip()}", original=tex)
        return result


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
