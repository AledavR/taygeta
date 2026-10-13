"""Estado de sesión de la UI NiceGUI (aislado por cliente del navegador).

NiceGUI no tiene el modelo de rerun de Streamlit: la página se construye una vez
por sesión y los eventos mutan objetos Python que refrescan los contenedores.
Este módulo mantiene un registro por ``client.id`` para que un recambio de
página (al volver a cargar la URL, p. ej.) no pierda el staging en curso; los
estados se liberan al desconectarse el cliente (ver ``app.on_disconnect`` en
``app.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from taygeta.ingest.pipeline import StagingResult


@dataclass
class QuestionState:
    """Estado de la revisión de una pregunta (figuras editadas, vistas…).

    ``figures`` es la fuente de verdad durante la revisión: refleja los
    borrados, recortes y altas que el usuario hace sobre las figuras
    estructuradas por el pipeline (mismo rol que ``st.session_state["figures_N"]``
    en la UI Streamlit).
    """

    figures: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SessionState:
    """Estado vivo de una sesión de revisión (por cliente del navegador)."""

    processed_path: str | None = None
    staging: StagingResult | None = None
    questions: dict[int, QuestionState] = field(default_factory=dict)
    #: Assets de figuras huérfanas (borradas o sin placeholder) pendientes de purga.
    orphans: dict[str, set[str]] = field(default_factory=dict)
    processing: bool = False


_sessions: dict[str, SessionState] = {}


def register(client_id: str) -> SessionState:
    """Devuelve (creándolo si no existe) el estado de sesión del cliente."""
    return _sessions.setdefault(client_id, SessionState())


def drop(client_id: str) -> None:
    """Libera el estado de una sesión al desconectarse su cliente."""
    _sessions.pop(client_id, None)
