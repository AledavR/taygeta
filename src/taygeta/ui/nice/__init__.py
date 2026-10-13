"""UI de revisión Human-in-the-Loop en NiceGUI.

Modelo de estado por eventos (sin reruns): la página se construye una vez por
sesión y los eventos mutan objetos Python y refrescan los contenedores.
"""

from __future__ import annotations

from nicegui import ui

__all__ = ["caption"]


def caption(text: str) -> None:
    """Equivalente a ``st.caption``: una etiqueta pequeña y gris."""
    ui.label(text).classes("text-xs text-gray-500")
