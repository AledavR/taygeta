"""Persistencia SQLite con búsqueda textual FTS5 para Taygeta.

Este módulo implementa la base de datos descrita en la sección 6 de la
especificación:

* ``questions`` con los campos exactos solicitados y ``payload_json`` con el
  objeto :class:`~taygeta.core.schema.EnrichedQuestion` serializado.
* ``questions_fts`` (virtual FTS5) sincronizada mediante triggers AFTER
  INSERT/UPDATE/DELETE, de modo que la búsqueda textual nunca queda desfasada.
* ``courses``, ``topics``, ``prerequisites`` y ``staging_sessions``.

La base de datos vive en ``DB_PATH`` (por defecto ``/app/data/taygeta.db``), es
decir, en el volumen persistente, nunca en la raíz del repositorio. La
inicialización del esquema es idempotente y :meth:`Database.status` informa del
estado real de la base de datos. No se almacenan soluciones ni respuestas
correctas: el ``payload_json`` está validado por esquemas que las prohíben.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from taygeta.core.embedding import cosine_similarity, decode_embedding, encode_embedding
from taygeta.core.schema import compute_canonical_hash, utc_now_iso
from taygeta.core.settings import Settings, get_settings

if TYPE_CHECKING:
    from taygeta.core.schema import EnrichedQuestion

__all__ = [
    "SCHEMA_VERSION",
    "STAGING_STATUSES",
    "Database",
    "DatabaseError",
    "get_database",
]

#: Versión lógica del esquema, registrada en ``schema_meta``.
SCHEMA_VERSION = "2"

#: Estados admitidos para una sesión de staging.
STAGING_STATUSES: tuple[str, ...] = ("VALID", "NEEDS_ATTENTION", "CONFIRMED", "DISCARDED")

#: Límite máximo de resultados para proteger el servicio RAG.
MAX_SEARCH_LIMIT = 100


class DatabaseError(RuntimeError):
    """Error controlado de acceso o integridad de la base de datos."""


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS courses (
    slug TEXT PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS topics (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    course_slug   TEXT NOT NULL REFERENCES courses(slug) ON DELETE CASCADE,
    topic_slug    TEXT NOT NULL,
    subtopic_slug TEXT
);

CREATE TABLE IF NOT EXISTS prerequisites (
    slug        TEXT PRIMARY KEY,
    description TEXT
);

CREATE TABLE IF NOT EXISTS questions (
    id                  TEXT PRIMARY KEY,
    course              TEXT NOT NULL,
    topic               TEXT NOT NULL,
    subtopic            TEXT,
    difficulty          INTEGER NOT NULL CHECK (difficulty BETWEEN 1 AND 5),
    prerequisites_json  TEXT NOT NULL DEFAULT '[]',
    payload_json        TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    embedding_json      TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS questions_fts USING fts5(
    id UNINDEXED,
    statement_raw,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS staging_sessions (
    session_id TEXT PRIMARY KEY,
    data_json  TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'NEEDS_ATTENTION'
);

CREATE INDEX IF NOT EXISTS idx_questions_course_topic ON questions(course, topic);
CREATE INDEX IF NOT EXISTS idx_questions_subtopic ON questions(subtopic);
CREATE INDEX IF NOT EXISTS idx_questions_difficulty ON questions(difficulty);
CREATE INDEX IF NOT EXISTS idx_questions_created_at ON questions(created_at);
CREATE INDEX IF NOT EXISTS idx_topics_course ON topics(course_slug);
CREATE INDEX IF NOT EXISTS idx_staging_status ON staging_sessions(status);
CREATE INDEX IF NOT EXISTS idx_staging_created_at ON staging_sessions(created_at);

-- El subtema puede ser NULL; COALESCE evita que varios NULL se consideren distintos.
CREATE UNIQUE INDEX IF NOT EXISTS uq_topics_course_topic_subtopic
    ON topics(course_slug, topic_slug, COALESCE(subtopic_slug, ''));

-- Sincronización FTS5 con questions mediante triggers.
CREATE TRIGGER IF NOT EXISTS questions_ai AFTER INSERT ON questions BEGIN
    INSERT INTO questions_fts(id, statement_raw)
    VALUES (new.id, json_extract(new.payload_json, '$.statement_raw'));
END;

CREATE TRIGGER IF NOT EXISTS questions_ad AFTER DELETE ON questions BEGIN
    DELETE FROM questions_fts WHERE id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS questions_au AFTER UPDATE ON questions BEGIN
    DELETE FROM questions_fts WHERE id = old.id;
    INSERT INTO questions_fts(id, statement_raw)
    VALUES (new.id, json_extract(new.payload_json, '$.statement_raw'));
END;
"""


def _build_match_query(query: str) -> str:
    """Convierte una consulta libre en una expresión MATCH segura para FTS5.

    Cada término se entrecomilla para evitar que la sintaxis de FTS5 (operadores
    ``AND``/``OR``/``NEAR``, comillas, paréntesis, ``*``) provoque errores o
    consultas inesperadas.
    """
    tokens = query.split()
    if not tokens:
        raise DatabaseError("la consulta de búsqueda no puede estar vacía")
    return " ".join('"' + token.replace('"', '""') + '"' for token in tokens)


class Database:
    """Acceso transaccional a la base de datos SQLite de Taygeta."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        settings: Settings | None = None,
        initialize: bool = True,
    ) -> None:
        """Crea el acceso a la base de datos.

        Args:
            db_path: Ruta del archivo SQLite. Si se omite, se usa ``DB_PATH``
                de la configuración.
            settings: Configuración a usar; por defecto la global.
            initialize: Si es ``True``, inicializa el esquema de forma
                idempotente al construir la instancia.
        """
        self.settings = settings or get_settings()
        resolved = db_path if db_path is not None else self.settings.database_path
        self.db_path = str(resolved)
        self._memory_connection: sqlite3.Connection | None = None
        self._fts5_enabled = False
        if initialize:
            self.initialize()

    # -- Conexión -----------------------------------------------------------
    @property
    def is_memory(self) -> bool:
        """Indica si la base de datos reside en memoria."""
        return self.db_path == ":memory:"

    def _new_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Contexto transaccional que confirma al salir y revierte ante error."""
        if self.is_memory:
            if self._memory_connection is None:
                self._memory_connection = self._new_connection()
            connection = self._memory_connection
            try:
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            return

        connection = self._new_connection()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    # -- Inicialización -----------------------------------------------------
    def initialize(self) -> None:
        """Crea el esquema de forma idempotente (seguro de llamar varias veces).

        Raises:
            DatabaseError: si la ruta no es accesible o el esquema no puede crearse.
        """
        try:
            if not self.is_memory:
                Path(self.db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            with self.connect() as connection:
                if not self.is_memory:
                    connection.execute("PRAGMA journal_mode = WAL")
                connection.executescript(_SCHEMA_SQL)
                self._migrate_schema(connection)
                connection.execute(
                    "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (SCHEMA_VERSION,),
                )
        except sqlite3.Error as exc:
            raise DatabaseError(
                f"no se pudo inicializar la base de datos en {self.db_path!r}: {exc}"
            ) from exc
        self._fts5_enabled = self._check_fts5()

    def _check_fts5(self) -> bool:
        try:
            with self.connect() as connection:
                connection.execute("SELECT count(*) FROM questions_fts").fetchone()
            return True
        except sqlite3.Error:
            return False

    @staticmethod
    def _migrate_schema(connection: sqlite3.Connection) -> None:
        """Aplica migraciones incrementales a bases de datos ya existentes.

        ``CREATE TABLE IF NOT EXISTS`` no altera tablas preexistentes, por lo que
        las columnas añadidas en versiones posteriores del esquema se crean aquí
        de forma idempotente (por ejemplo ``questions.embedding_json``).
        """
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(questions)").fetchall()
        }
        if "embedding_json" not in columns:
            connection.execute("ALTER TABLE questions ADD COLUMN embedding_json TEXT")

    def rebuild_fts(self) -> int:
        """Regenera ``questions_fts`` desde ``questions`` y devuelve el número de filas."""
        with self.connect() as connection:
            connection.execute("DELETE FROM questions_fts")
            connection.execute(
                "INSERT INTO questions_fts(id, statement_raw) "
                "SELECT id, json_extract(payload_json, '$.statement_raw') FROM questions"
            )
            row = connection.execute("SELECT count(*) AS c FROM questions_fts").fetchone()
        return int(row["c"])

    # -- Preguntas ----------------------------------------------------------
    def question_exists(self, question_id: str) -> bool:
        """Comprueba si existe una pregunta con el identificador indicado."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM questions WHERE id = ? LIMIT 1", (question_id,)
            ).fetchone()
        return row is not None

    def insert_question(
        self, question: EnrichedQuestion, *, embedding: Sequence[float] | None = None
    ) -> str:
        """Inserta una pregunta nueva y devuelve el estado de la inserción.

        En entorno ``local`` calcula el ``canonical_hash`` del enunciado y, si ya
        existe una pregunta con el mismo hash en el mismo curso, bloquea la
        inserción devolviendo ``"DUPLICATE"``. El ``embedding`` (opcional) se
        almacena para la deduplicación semántica posterior.

        Raises:
            DatabaseError: si ya existe una pregunta con el mismo ``id``.
        """
        if self.settings.APP_ENV == "local":
            question = question.model_copy(
                update={"canonical_hash": compute_canonical_hash(question.statement_raw)}
            )
        values = self._question_values(question, embedding)
        with self.connect() as connection:
            if self.settings.APP_ENV == "local":
                duplicate = connection.execute(
                    "SELECT 1 FROM questions "
                    "WHERE course = ? AND json_extract(payload_json, '$.canonical_hash') = ? "
                    "LIMIT 1",
                    (question.course, question.canonical_hash),
                ).fetchone()
                if duplicate is not None:
                    return "DUPLICATE"
            try:
                connection.execute(
                    "INSERT INTO questions("
                    "id, course, topic, subtopic, difficulty, prerequisites_json, "
                    "payload_json, created_at, embedding_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    values,
                )
            except sqlite3.IntegrityError as exc:
                raise DatabaseError(f"ya existe una pregunta con id {question.id!r}") from exc
        return "INSERTED"

    def update_question(
        self, question: EnrichedQuestion, *, embedding: Sequence[float] | None = None
    ) -> None:
        """Actualiza una pregunta existente y sincroniza FTS5.

        Si ``embedding`` es ``None``, el embedding almacenado se conserva tal cual
        (no se borra); para vaciarlo explícitamente pásese una lista vacía.

        Raises:
            DatabaseError: si la pregunta no existe.
        """
        values = self._question_values(question, embedding)
        with self.connect() as connection:
            if embedding is None:
                cursor = connection.execute(
                    "UPDATE questions SET course = ?, topic = ?, subtopic = ?, difficulty = ?, "
                    "prerequisites_json = ?, payload_json = ?, created_at = ? WHERE id = ?",
                    (
                        values[1],
                        values[2],
                        values[3],
                        values[4],
                        values[5],
                        values[6],
                        values[7],
                        values[0],
                    ),
                )
            else:
                cursor = connection.execute(
                    "UPDATE questions SET course = ?, topic = ?, subtopic = ?, difficulty = ?, "
                    "prerequisites_json = ?, payload_json = ?, created_at = ?, embedding_json = ? "
                    "WHERE id = ?",
                    (
                        values[1],
                        values[2],
                        values[3],
                        values[4],
                        values[5],
                        values[6],
                        values[7],
                        values[8],
                        values[0],
                    ),
                )
            if cursor.rowcount == 0:
                raise DatabaseError(f"no existe una pregunta con id {question.id!r}")

    def upsert_question(
        self, question: EnrichedQuestion, *, embedding: Sequence[float] | None = None
    ) -> None:
        """Inserta o reemplaza una pregunta manteniendo FTS5 coherente."""
        values = self._question_values(question, embedding)
        with self.connect() as connection:
            connection.execute("DELETE FROM questions WHERE id = ?", (question.id,))
            connection.execute(
                "INSERT INTO questions("
                "id, course, topic, subtopic, difficulty, prerequisites_json, "
                "payload_json, created_at, embedding_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )

    def get_question(self, question_id: str) -> EnrichedQuestion | None:
        """Recupera una pregunta validada o ``None`` si no existe."""
        from taygeta.core.schema import EnrichedQuestion

        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM questions WHERE id = ?", (question_id,)
            ).fetchone()
        if row is None:
            return None
        return EnrichedQuestion.model_validate_json(row["payload_json"])

    def delete_question(self, question_id: str) -> bool:
        """Elimina una pregunta (y su fila FTS5 vía trigger). Devuelve si existía."""
        with self.connect() as connection:
            cursor = connection.execute("DELETE FROM questions WHERE id = ?", (question_id,))
        return cursor.rowcount > 0

    def count_questions(self) -> int:
        """Devuelve el número total de preguntas almacenadas."""
        with self.connect() as connection:
            row = connection.execute("SELECT count(*) AS c FROM questions").fetchone()
        return int(row["c"])

    def list_question_ids(self) -> list[str]:
        """Devuelve los identificadores de todas las preguntas, ordenados por id."""
        with self.connect() as connection:
            rows = connection.execute("SELECT id FROM questions ORDER BY id").fetchall()
        return [row["id"] for row in rows]

    def list_questions(
        self,
        *,
        course: str | None = None,
        topic: str | None = None,
        subtopic: str | None = None,
        difficulty: int | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[EnrichedQuestion]:
        """Lista preguntas aplicando filtros curriculares (consultas parametrizadas)."""
        from taygeta.core.schema import EnrichedQuestion

        conditions: list[str] = []
        params: list[Any] = []
        if course is not None:
            conditions.append("course = ?")
            params.append(course)
        if topic is not None:
            conditions.append("topic = ?")
            params.append(topic)
        if subtopic is not None:
            conditions.append("subtopic = ?")
            params.append(subtopic)
        if difficulty is not None:
            conditions.append("difficulty = ?")
            params.append(difficulty)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = (
            f"SELECT payload_json FROM questions {where} "
            "ORDER BY created_at DESC, id ASC LIMIT ? OFFSET ?"
        )
        params.extend([self._clamp_limit(limit), max(0, offset)])
        with self.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [EnrichedQuestion.model_validate_json(row["payload_json"]) for row in rows]

    def search_questions(
        self,
        *,
        query: str | None = None,
        course: str | None = None,
        topic: str | None = None,
        subtopic: str | None = None,
        difficulty: int | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[EnrichedQuestion]:
        """Recupera preguntas por texto (FTS5) y/o filtros curriculares.

        Usa consultas parametrizadas y un límite acotado. Nunca devuelve
        soluciones ni respuestas correctas.
        """
        from taygeta.core.schema import EnrichedQuestion

        conditions: list[str] = []
        params: list[Any] = []

        if query:
            source = (
                "FROM questions_fts JOIN questions q ON q.id = questions_fts.id "
                "WHERE questions_fts MATCH ?"
            )
            params.append(_build_match_query(query))
            order = "ORDER BY bm25(questions_fts), q.created_at DESC"
        else:
            source = "FROM questions q WHERE 1 = 1"
            order = "ORDER BY q.created_at DESC, q.id ASC"

        if course is not None:
            conditions.append("q.course = ?")
            params.append(course)
        if topic is not None:
            conditions.append("q.topic = ?")
            params.append(topic)
        if subtopic is not None:
            conditions.append("q.subtopic = ?")
            params.append(subtopic)
        if difficulty is not None:
            conditions.append("q.difficulty = ?")
            params.append(difficulty)

        extra = f" AND {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT q.payload_json {source}{extra} {order} LIMIT ? OFFSET ?"
        params.extend([self._clamp_limit(limit), max(0, offset)])

        with self.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [EnrichedQuestion.model_validate_json(row["payload_json"]) for row in rows]

    # -- Taxonomía ----------------------------------------------------------
    def upsert_course(self, slug: str, name: str) -> None:
        """Inserta o actualiza un curso del catálogo."""
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO courses(slug, name) VALUES (?, ?) "
                "ON CONFLICT(slug) DO UPDATE SET name = excluded.name",
                (slug, name),
            )

    def get_course_name(self, slug: str) -> str | None:
        """Devuelve el nombre legible de un curso, o ``None`` si no está catalogado."""
        with self.connect() as connection:
            row = connection.execute("SELECT name FROM courses WHERE slug = ?", (slug,)).fetchone()
        return row["name"] if row else None

    def upsert_topic(
        self,
        course_slug: str,
        topic_slug: str,
        subtopic_slug: str | None = None,
    ) -> int:
        """Inserta si falta y devuelve el identificador del tema/subtema."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id FROM topics WHERE course_slug = ? AND topic_slug = ? "
                "AND COALESCE(subtopic_slug, '') = COALESCE(?, '')",
                (course_slug, topic_slug, subtopic_slug),
            ).fetchone()
            if row is not None:
                return int(row["id"])
            try:
                cursor = connection.execute(
                    "INSERT INTO topics(course_slug, topic_slug, subtopic_slug) VALUES (?, ?, ?)",
                    (course_slug, topic_slug, subtopic_slug),
                )
            except sqlite3.IntegrityError as exc:
                raise DatabaseError(
                    f"no se pudo crear el tema {topic_slug!r}: el curso {course_slug!r} no existe"
                ) from exc
            return int(cursor.lastrowid or 0)

    def upsert_prerequisite(self, slug: str, description: str | None = None) -> None:
        """Inserta o actualiza un prerrequisito del catálogo."""
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO prerequisites(slug, description) VALUES (?, ?) "
                "ON CONFLICT(slug) DO UPDATE SET description = COALESCE(excluded.description, "
                "prerequisites.description)",
                (slug, description),
            )

    def add_prerequisites(self, slugs: list[str]) -> list[str]:
        """Añade dinámicamente nuevos tags de prerrequisitos al catálogo."""
        slugs = list(dict.fromkeys(slug for slug in slugs if slug))
        with self.connect() as connection:
            connection.executemany(
                "INSERT INTO prerequisites(slug, description) VALUES (?, NULL) "
                "ON CONFLICT(slug) DO NOTHING",
                [(slug,) for slug in slugs],
            )
        return slugs

    def list_courses(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT slug, name FROM courses ORDER BY slug").fetchall()
        return [dict(row) for row in rows]

    def list_topics(self, course_slug: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as connection:
            if course_slug is None:
                rows = connection.execute(
                    "SELECT id, course_slug, topic_slug, subtopic_slug FROM topics "
                    "ORDER BY course_slug, topic_slug, subtopic_slug"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT id, course_slug, topic_slug, subtopic_slug FROM topics "
                    "WHERE course_slug = ? ORDER BY topic_slug, subtopic_slug",
                    (course_slug,),
                ).fetchall()
        return [dict(row) for row in rows]

    def list_prerequisites(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT slug, description FROM prerequisites ORDER BY slug"
            ).fetchall()
        return [dict(row) for row in rows]

    def list_taxonomy(self) -> dict[str, Any]:
        """Devuelve el catálogo completo de cursos, temas, subtemas y prerrequisitos."""
        return {
            "courses": self.list_courses(),
            "topics": self.list_topics(),
            "prerequisites": self.list_prerequisites(),
        }

    def get_live_taxonomy_tree(self) -> dict[str, dict[str, list[str]]]:
        """Devuelve la taxonomía curricular viva de la base de datos.

        Reconstruye la jerarquía ``curso -> tema -> subtemas`` realmente en uso a
        partir de las preguntas ya confirmadas. Solo se incluyen cursos y temas
        presentes en la tabla ``questions``, de modo que el LLM pueda reutilizar
        los slugs existentes en lugar de inventar variantes. El diccionario no
        almacena soluciones ni respuestas correctas: únicamente slugs curriculares.
        """
        tree: dict[str, dict[str, list[str]]] = {}
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT course, topic, subtopic FROM questions "
                "WHERE course IS NOT NULL ORDER BY course, topic, subtopic"
            ).fetchall()
        for row in rows:
            course = row["course"]
            topic = row["topic"]
            subtopic = row["subtopic"]
            topics = tree.setdefault(course, {})
            subtopics = topics.setdefault(topic, [])
            if subtopic and subtopic not in subtopics:
                subtopics.append(subtopic)
        return tree

    # -- Deduplicación semántica -------------------------------------------
    def get_embedding(self, question_id: str) -> list[float] | None:
        """Devuelve el embedding almacenado de una pregunta, o ``None`` si no tiene."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT embedding_json FROM questions WHERE id = ?", (question_id,)
            ).fetchone()
        if row is None or not row["embedding_json"]:
            return None
        return decode_embedding(row["embedding_json"])

    def list_embeddings(self, *, course: str | None = None) -> list[tuple[str, list[float]]]:
        """Lista ``(id, embedding)`` de las preguntas que ya tienen vector.

        Se usa para calcular la similitud coseno en memoria (el volumen actual es
        bajo). Si se indica ``course``, se restringe a ese curso.
        """
        if course is not None:
            sql = (
                "SELECT id, embedding_json FROM questions "
                "WHERE course = ? AND embedding_json IS NOT NULL ORDER BY id"
            )
            params: tuple[Any, ...] = (course,)
        else:
            sql = (
                "SELECT id, embedding_json FROM questions "
                "WHERE embedding_json IS NOT NULL ORDER BY id"
            )
            params = ()
        with self.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        result: list[tuple[str, list[float]]] = []
        for row in rows:
            vector = decode_embedding(row["embedding_json"])
            if vector:
                result.append((row["id"], vector))
        return result

    def find_similar_questions(
        self,
        embedding: Sequence[float],
        *,
        threshold: float,
        course: str | None = None,
        limit: int = 5,
    ) -> list[tuple[str, float]]:
        """Devuelve las preguntas con similitud coseno >= ``threshold``.

        Ordena por similitud descendente y acota a ``limit`` resultados. Compara
        solo contra preguntas que ya tienen embedding almacenado.
        """
        matches: list[tuple[str, float]] = []
        for question_id, vector in self.list_embeddings(course=course):
            similarity = cosine_similarity(embedding, vector)
            if similarity >= threshold:
                matches.append((question_id, similarity))
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches[:limit]

    # -- Staging ------------------------------------------------------------
    def create_staging_session(
        self,
        data: dict[str, Any] | list[Any] | str,
        *,
        status: str = "NEEDS_ATTENTION",
        session_id: str | None = None,
    ) -> str:
        """Crea una sesión de staging y devuelve su identificador."""
        self._validate_status(status)
        sid = session_id or uuid.uuid4().hex
        data_json = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO staging_sessions(session_id, data_json, created_at, status) "
                "VALUES (?, ?, ?, ?)",
                (sid, data_json, utc_now_iso(), status),
            )
        return sid

    def get_staging_session(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT session_id, data_json, created_at, status FROM staging_sessions "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        session = dict(row)
        session["data"] = json.loads(session.pop("data_json"))
        return session

    def update_staging_session(
        self,
        session_id: str,
        *,
        data: dict[str, Any] | list[Any] | str | None = None,
        status: str | None = None,
    ) -> bool:
        """Actualiza los datos y/o el estado de una sesión de staging."""
        if data is None and status is None:
            raise DatabaseError("no se indicó ningún cambio para la sesión de staging")
        if status is not None:
            self._validate_status(status)

        assignments: list[str] = []
        params: list[Any] = []
        if data is not None:
            assignments.append("data_json = ?")
            params.append(data if isinstance(data, str) else json.dumps(data, ensure_ascii=False))
        if status is not None:
            assignments.append("status = ?")
            params.append(status)
        params.append(session_id)

        with self.connect() as connection:
            cursor = connection.execute(
                f"UPDATE staging_sessions SET {', '.join(assignments)} WHERE session_id = ?",
                params,
            )
        return cursor.rowcount > 0

    def delete_staging_session(self, session_id: str) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM staging_sessions WHERE session_id = ?", (session_id,)
            )
        return cursor.rowcount > 0

    def list_staging_sessions(self, status: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as connection:
            if status is None:
                rows = connection.execute(
                    "SELECT session_id, created_at, status FROM staging_sessions "
                    "ORDER BY created_at DESC"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT session_id, created_at, status FROM staging_sessions "
                    "WHERE status = ? ORDER BY created_at DESC",
                    (status,),
                ).fetchall()
        return [dict(row) for row in rows]

    # -- Estado -------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """Devuelve un resumen verificable del estado de la base de datos."""
        with self.connect() as connection:
            tables = [
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                ).fetchall()
            ]
            counts: dict[str, int] = {}
            for table in ("questions", "courses", "topics", "prerequisites", "staging_sessions"):
                counts[table] = int(
                    connection.execute(f"SELECT count(*) AS c FROM {table}").fetchone()["c"]
                )
            version_row = connection.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        return {
            "db_path": self.db_path,
            "schema_version": version_row["value"] if version_row else None,
            "fts5_enabled": self._fts5_enabled,
            "integrity": integrity,
            "tables": tables,
            "counts": counts,
        }

    # -- Utilidades internas ------------------------------------------------
    @staticmethod
    def _clamp_limit(limit: int) -> int:
        if limit < 1:
            return 1
        return min(limit, MAX_SEARCH_LIMIT)

    @staticmethod
    def _validate_status(status: str) -> None:
        if status not in STAGING_STATUSES:
            raise DatabaseError(
                f"estado de staging inválido {status!r}; use uno de {STAGING_STATUSES}"
            )

    @staticmethod
    def _question_values(
        question: EnrichedQuestion, embedding: Sequence[float] | None = None
    ) -> tuple[Any, ...]:
        return (
            question.id,
            question.course,
            question.topic,
            question.subtopic,
            question.difficulty,
            json.dumps(question.prerequisites, ensure_ascii=False),
            question.model_dump_json(),
            question.created_at,
            encode_embedding(embedding),
        )


def get_database() -> Database:
    """Devuelve una instancia de :class:`Database` sobre la ruta efectiva.

    Usa ``DATABASE_URL`` si está configurada; en caso contrario ``DB_PATH``.
    """
    return Database(get_settings().database_path)
