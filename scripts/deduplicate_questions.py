#!/usr/bin/env python3
"""Deduplicación semántica de preguntas ya almacenadas (migración puntual).

Este script:

1. genera el embedding de cada pregunta que aún no lo tenga (con
   ``LLM_EMBEDDING_MODEL``) y lo persiste en ``questions.embedding_json``;
2. calcula la similitud coseno entre todas las preguntas del mismo curso;
3. para cada par cuya similitud supere el umbral, consulta (opcionalmente) al
   juez LLM para confirmar que son matemáticamente la misma pregunta;
4. fusiona los duplicados confirmados: conserva la pregunta más antigua, añade
   los ``source_reference`` y prerrequisitos de la nueva y elimina la fila
   redundante.

Nunca se descartan soluciones ni respuestas correctas (la base no las contiene).

Uso:

    uv run python scripts/deduplicate_questions.py --db /ruta/a/taygeta.db
    uv run python scripts/deduplicate_questions.py --db /ruta/a/taygeta.db --dry-run
    uv run python scripts/deduplicate_questions.py --threshold 0.95 --no-judge --db taygeta.db
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict

from taygeta.agents.embeddings import Embedder, question_embedding_text
from taygeta.core.embedding import cosine_similarity
from taygeta.core.schema import EnrichedQuestion
from taygeta.server.db import Database
from taygeta.server.deduplication import DeduplicationService, merge_duplicate_metadata

logger = logging.getLogger("deduplicate_questions")


def _resolve_db_path(explicit: str | None) -> str:
    """Resuelve la ruta de la base de datos: argumento explícito o configuración."""
    if explicit:
        return explicit
    try:
        from taygeta.core.settings import get_settings

        return get_settings().database_path
    except Exception:  # pragma: no cover - sin configuración disponible
        return "/app/data/taygeta.db"


def _load_questions(database: Database) -> list[EnrichedQuestion]:
    """Carga todas las preguntas de la base, ordenadas por fecha de creación."""
    questions: list[EnrichedQuestion] = []
    for question_id in database.list_question_ids():
        question = database.get_question(question_id)
        if question is not None:
            questions.append(question)
    questions.sort(key=lambda q: (q.created_at, q.id))
    return questions


def backfill_embeddings(database: Database, embedder: Embedder, *, dry_run: bool = False) -> int:
    """Genera y persiste los embeddings que falten. Devuelve cuántos creó."""
    questions = _load_questions(database)
    missing = [q for q in questions if not database.get_embedding(q.id)]
    if not missing:
        return 0

    texts = [question_embedding_text(q) for q in missing]
    try:
        vectors = embedder.embed_many(texts)
    except Exception as exc:  # noqa: BLE001 - la migración no debe abortar por el proveedor
        logger.warning("no se pudieron generar embeddings por lotes: %s", exc)
        vectors = []
        for text in texts:
            try:
                vectors.append(embedder.embed(text))
            except Exception as item_exc:  # noqa: BLE001 - degrada por elemento
                logger.warning("embedding individual omitido: %s", item_exc)
                vectors.append([])

    created = 0
    for question, vector in zip(missing, vectors, strict=True):
        if not vector:
            continue
        if not dry_run:
            database.update_question(question, embedding=vector)
        created += 1
    return created


def deduplicate(
    database: Database,
    *,
    threshold: float,
    use_judge: bool,
    dry_run: bool = False,
) -> int:
    """Fusiona duplicados semánticos y devuelve cuántos fusionó.

    Compara dentro del mismo curso y conserva la pregunta más antigua.
    """
    service = DeduplicationService(database, threshold=threshold)
    questions = _load_questions(database)

    by_course: dict[str, list[EnrichedQuestion]] = defaultdict(list)
    for question in questions:
        by_course[question.course].append(question)

    merged_ids: set[str] = set()
    merged = 0
    for course_questions in by_course.values():
        for index, question in enumerate(course_questions):
            if question.id in merged_ids:
                continue
            embedding = database.get_embedding(question.id)
            if not embedding:
                continue
            for other in course_questions[index + 1 :]:
                if other.id in merged_ids:
                    continue
                other_embedding = database.get_embedding(other.id)
                if not other_embedding:
                    continue
                similarity = cosine_similarity(embedding, other_embedding)
                if similarity < threshold:
                    continue
                if not _is_confirmed_duplicate(service, question, other, use_judge):
                    continue
                merged_question = merge_duplicate_metadata(question, other)
                if not dry_run:
                    database.update_question(merged_question, embedding=embedding)
                    database.delete_question(other.id)
                merged += 1
                merged_ids.add(other.id)
    return merged


def _is_confirmed_duplicate(
    service: DeduplicationService,
    question: EnrichedQuestion,
    other: EnrichedQuestion,
    use_judge: bool,
) -> bool:
    """Confirma (por juez LLM o por umbral) que ``other`` es duplicado de ``question``."""
    if not use_judge:
        return True
    judgement = service.judge(question, [other])
    if judgement is None:
        # Juez no disponible o falló: se respeta el umbral de similitud.
        return True
    return judgement.is_duplicate and judgement.matching_index == 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Genera embeddings y fusiona preguntas duplicadas semánticamente."
    )
    parser.add_argument("--db", default=None, help="Ruta del archivo SQLite.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Umbral de similitud coseno (por defecto DUPLICATE_SIMILARITY_THRESHOLD).",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="No consultar al juez LLM; fusionar solo por umbral de similitud.",
    )
    parser.add_argument(
        "--no-backfill",
        action="store_true",
        help="No generar embeddings para las preguntas que aún no los tienen.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo informa de las fusiones y embeddings que se harían, sin escribir.",
    )
    args = parser.parse_args(argv)

    db_path = _resolve_db_path(args.db)
    try:
        database = Database(db_path)
        settings = database.settings
        threshold = (
            args.threshold
            if args.threshold is not None
            else settings.DUPLICATE_SIMILARITY_THRESHOLD
        )
        if not 0.0 <= threshold <= 1.0:
            print("el umbral debe estar entre 0.0 y 1.0", file=sys.stderr)
            raise SystemExit(2)

        created = 0
        if not args.no_backfill:
            created = backfill_embeddings(database, Embedder(settings), dry_run=args.dry_run)

        merged = deduplicate(
            database,
            threshold=threshold,
            use_judge=not args.no_judge,
            dry_run=args.dry_run,
        )
    except Exception as exc:  # noqa: BLE001 - se informa y se sale con error
        print(f"error al procesar la base de datos {db_path!r}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    action = "se harían" if args.dry_run else "se hicieron"
    print(
        f"{created} embedding(s) generado(s) y {merged} pregunta(s) duplicada(s) "
        f"fusionada(s) {action} en {db_path} (umbral={threshold}, juez="
        f"{'no' if args.no_judge else 'sí'})."
    )


if __name__ == "__main__":
    main()
