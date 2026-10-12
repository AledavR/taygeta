#!/usr/bin/env python3
"""Migración puntual: limpieza de barras de norma ``\\|`` en ``statement_raw``.

Los modelos OCR/LLM a veces emiten ``\\|`` (barra de norma LaTeX) en lugar de
``||`` (norma Typst). El transpilador rechaza el comando ``\\|`` y la compilación
del examen falla. Este script recorre las preguntas de ``questions`` y reescribe
``\\|`` → ``||`` en ``payload_json.statement_raw`` y en cada
``payload_json.choices[].content_raw``.

La tabla ``questions_fts`` se mantiene sincronizada automáticamente: el trigger
``questions_au`` vuelve a extraer ``statement_raw`` del ``payload_json`` al
actualizar la fila.

Uso:

    uv run python scripts/migrate_norm_bars.py --db /ruta/a/taygeta.db
    uv run python scripts/migrate_norm_bars.py --db /ruta/a/taygeta.db --dry-run
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys

from taygeta.core.schema import sanitize_norm_bars


def _resolve_db_path(explicit: str | None) -> str:
    """Resuelve la ruta de la base de datos: argumento explícito o configuración."""
    if explicit:
        return explicit
    try:
        from taygeta.core.settings import get_settings

        return get_settings().database_path
    except Exception:  # pragma: no cover - sin configuración disponible
        return "/app/data/taygeta.db"


def migrate(db_path: str, *, dry_run: bool = False) -> int:
    """Reescribe ``\\|`` → ``||`` en los enunciados y alternativas almacenados.

    Devuelve el número de filas modificadas.
    """
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute("SELECT id, payload_json FROM questions").fetchall()
        changed = 0
        for row in rows:
            payload = json.loads(row["payload_json"])
            before = json.dumps(payload, ensure_ascii=False, sort_keys=True)

            statement = payload.get("statement_raw")
            if isinstance(statement, str):
                payload["statement_raw"] = sanitize_norm_bars(statement)
            for choice in payload.get("choices") or []:
                content = choice.get("content_raw") if isinstance(choice, dict) else None
                if isinstance(content, str):
                    choice["content_raw"] = sanitize_norm_bars(content)

            after = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            if after == before:
                continue
            changed += 1
            if not dry_run:
                connection.execute(
                    "UPDATE questions SET payload_json = ? WHERE id = ?",
                    (json.dumps(payload, ensure_ascii=False), row["id"]),
                )
        if not dry_run:
            connection.commit()
        return changed
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Limpia barras de norma \\| en statement_raw.")
    parser.add_argument("--db", default=None, help="Ruta del archivo SQLite.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo informa de las filas que se modificarían, sin escribir.",
    )
    args = parser.parse_args(argv)

    db_path = _resolve_db_path(args.db)
    try:
        changed = migrate(db_path, dry_run=args.dry_run)
    except sqlite3.Error as exc:
        print(f"error al acceder a la base de datos {db_path!r}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    action = "modificarían" if args.dry_run else "modificadas"
    print(f"{changed} pregunta(s) {action} en {db_path}")


if __name__ == "__main__":
    main()
