#!/usr/bin/env python3
"""Migración puntual: canonicalización y deduplicación de prerrequisitos.

Los modelos OCR/LLM pueden emitir variantes de slug (singular/plural o sinónimos)
para los prerrequisitos. Este script recorre las preguntas de ``questions`` y:

1. resuelve cada prerrequisito contra :data:`PREREQUISITE_ALIASES` (aliasing canónico);
2. deduplica los slugs resultantes preservando el orden;
3. actualiza ``prerequisites_json`` y ``payload_json.prerequisites`` de cada fila;
4. normaliza también el catálogo ``prerequisites`` (inserta el slug canónico si
   falta y elimina las filas que son alias), de modo que ``list_taxonomy`` quede
   coherente.

Uso:

    uv run python scripts/migrate_prerequisites.py --db /ruta/a/taygeta.db
    uv run python scripts/migrate_prerequisites.py --db /ruta/a/taygeta.db --dry-run
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys

from taygeta.domain.taxonomy import canonical_prerequisite, canonicalize_prerequisites


def _resolve_db_path(explicit: str | None) -> str:
    """Resuelve la ruta de la base de datos: argumento explícito o configuración."""
    if explicit:
        return explicit
    try:
        from taygeta.core.settings import get_settings

        return get_settings().database_path
    except Exception:  # pragma: no cover - sin configuración disponible
        return "/app/data/taygeta.db"


def migrate(db_path: str, *, dry_run: bool = False) -> tuple[int, int]:
    """Canonicaliza los prerrequisitos de preguntas y catálogo.

    Devuelve ``(preguntas_modificadas, catalogos_modificados)``.
    """
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        # 1) Preguntas: prerequisites_json + payload_json.prerequisites.
        rows = connection.execute(
            "SELECT id, prerequisites_json, payload_json FROM questions"
        ).fetchall()
        questions_changed = 0
        for row in rows:
            try:
                stored = json.loads(row["prerequisites_json"] or "[]")
            except json.JSONDecodeError:
                stored = []
            if not isinstance(stored, list):
                stored = []
            original = [str(slug) for slug in stored if isinstance(slug, str)]
            canonical = canonicalize_prerequisites(original)

            payload = json.loads(row["payload_json"])
            payload_prereqs = payload.get("prerequisites")
            if canonical == original and payload_prereqs == canonical:
                continue

            payload["prerequisites"] = canonical
            questions_changed += 1
            if not dry_run:
                connection.execute(
                    "UPDATE questions SET prerequisites_json = ?, payload_json = ? WHERE id = ?",
                    (
                        json.dumps(canonical, ensure_ascii=False),
                        json.dumps(payload, ensure_ascii=False),
                        row["id"],
                    ),
                )

        # 2) Catálogo de prerrequisitos: resolver alias → canónico.
        catalog_rows = connection.execute("SELECT slug, description FROM prerequisites").fetchall()
        catalog_changed = 0
        for row in catalog_rows:
            slug = row["slug"]
            canonical = canonical_prerequisite(slug)
            if canonical == slug:
                continue
            connection.execute(
                "INSERT INTO prerequisites(slug, description) VALUES (?, ?) "
                "ON CONFLICT(slug) DO UPDATE SET description = "
                "COALESCE(prerequisites.description, excluded.description)",
                (canonical, row["description"]),
            )
            if not dry_run:
                connection.execute("DELETE FROM prerequisites WHERE slug = ?", (slug,))
            catalog_changed += 1

        if not dry_run:
            connection.commit()
        return questions_changed, catalog_changed
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Canonicaliza los prerrequisitos almacenados.")
    parser.add_argument("--db", default=None, help="Ruta del archivo SQLite.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo informa de las filas que se modificarían, sin escribir.",
    )
    args = parser.parse_args(argv)

    db_path = _resolve_db_path(args.db)
    try:
        questions_changed, catalog_changed = migrate(db_path, dry_run=args.dry_run)
    except sqlite3.Error as exc:
        print(f"error al acceder a la base de datos {db_path!r}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    action = "modificarían" if args.dry_run else "modificadas"
    print(
        f"{questions_changed} pregunta(s) y {catalog_changed} fila(s) de catálogo {action} "
        f"en {db_path}"
    )


if __name__ == "__main__":
    main()
