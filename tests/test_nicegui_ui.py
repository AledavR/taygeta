"""Pruebas del port NiceGUI de la revisión HITL (arnés ``user_simulation``).

Estas pruebas construyen la página real sin navegador (``httpx`` + ``socketio``
en proceso). Notas importantes:

* ``user_simulation`` **solo funciona bajo pytest**: el reset de globals llama a
  ``App.reset()`` -> ``Storage.clear()``, que accede a ``context.client`` salvo
  cuando ``helpers.is_pytest()`` es verdadero. Ejecutarlo como script suelto
  activa el "script mode" y rompe la app basada en ``@ui.page``.
* Playwright no es necesario (``screen``/Selenium sí lo requerirían).
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from nicegui import ui

from taygeta.core.schema import Choice, EnrichedQuestion, FigureReference, utc_now_iso
from taygeta.core.settings import get_settings
from taygeta.ingest.pipeline import StagedQuestion
from taygeta.ingest.validator import STATUS_NEEDS_ATTENTION, STATUS_VALID
from taygeta.server.db import Database
from taygeta.ui.nice.question_panel import QuestionPanel
from taygeta.ui.nice.state import QuestionState, SessionState

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Fuerza el backend asyncio del plugin anyio."""
    return "asyncio"


@pytest.fixture
def ui_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Aísla rutas/persistencia de la UI en un directorio temporal."""
    for name in ("uploads", "downloads"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DB_PATH", str(tmp_path / "taygeta.db"))
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("DOWNLOADS_DIR", str(tmp_path / "downloads"))
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://localhost:8333")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class _StubStorage:
    """Almacenamiento nulo: evita cualquier acceso de red al cargar figuras."""

    bucket = "taygeta-assets"

    def get_bytes(self, key: str, bucket: str | None = None) -> bytes | None:
        return None


class _FakeStorage:
    """Almacenamiento en memoria para probar el alta/recorte de figuras sin red."""

    bucket = "taygeta-assets"

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def get_bytes(self, key: str, bucket: str | None = None) -> bytes | None:
        return self.objects.get(key)

    def store_asset_bytes(
        self,
        data: bytes,
        *,
        prefix: str = "figures",
        asset_id: str | None = None,
        extension: str = ".png",
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        key = f"{prefix}/{asset_id or uuid4().hex}{extension}"
        self.objects[key] = data
        return f"s3://{self.bucket}/{key}"

    def relative_path(self, reference: str) -> str:
        key = reference.split("/", 3)[3]
        return f"/buckets/{self.bucket}/{key}"


def _make_questions() -> list[StagedQuestion]:
    valid = EnrichedQuestion(
        id=str(uuid4()),
        statement_raw="Resuelve $x^2 - 1 = 0$ y elige la alternativa. {{fig:1}}",
        question_type="multiple_choice",
        choices=[Choice(id="A", content_raw="$x=1$"), Choice(id="B", content_raw="$x=-1$")],
        figures=[
            FigureReference(
                id="fig:1",
                page=0,
                bbox=[100, 100, 400, 400],
                asset_path="/buckets/taygeta-assets/figures/x.png",
            )
        ],
        course="algebra",
        topic="ecuaciones",
        subtopic="cuadraticas",
        difficulty=2,
        prerequisites=["factorizacion"],
        created_at=utc_now_iso(),
    )
    attention = EnrichedQuestion(
        id=str(uuid4()),
        statement_raw="Demuestra que la suma de ángulos internos de un triángulo es 180°.",
        question_type="open_ended",
        choices=None,
        course="geometria",
        topic="triangulos",
        subtopic=None,
        difficulty=3,
        prerequisites=[],
        created_at=utc_now_iso(),
    )
    return [
        StagedQuestion(question=valid, status=STATUS_VALID, reasons=[]),
        StagedQuestion(
            question=attention,
            status=STATUS_NEEDS_ATTENTION,
            reasons=["validación semántica incompleta"],
        ),
    ]


async def test_index_page_builds(ui_env: None, tmp_path: Path) -> None:
    """La página principal del port NiceGUI se construye sin errores."""
    from nicegui.testing import user_simulation

    launcher = tmp_path / "niceui_launcher.py"
    launcher.write_text("from taygeta.ui.nice.app import main\n\nmain()\n")

    async with user_simulation(main_file=launcher) as user:
        await user.open("/")
        await user.should_see(content="Taygeta — Ingesta y revisión Human-in-the-Loop")
        await user.should_see("1. Subir documento")
        await user.should_see("2. Procesar")
        await user.should_see("3. Revisión y confirmación")

        # Modo oscuro: el botón flotante alterna el estado del cliente.
        user.find("toggle-modo-oscuro").click()
        dark = next(
            element
            for element in user.find(ui.dark_mode).elements
            if isinstance(element, ui.dark_mode)
        )
        assert dark.value is True


async def test_question_panels_build_and_switch_view(ui_env: None) -> None:
    """Los paneles se construyen con datos falsos y responden a eventos."""
    settings = get_settings()
    database = Database(settings=settings)
    staged = _make_questions()

    def root() -> None:
        session = SessionState()
        session.questions = {index: QuestionState() for index in range(len(staged))}
        for index, item in enumerate(staged):
            QuestionPanel(
                ui.column().classes("w-full"),
                index=index,
                staged=item,
                session=session,
                source_path=None,
                storage=_StubStorage(),
                settings=settings,
                database=database,
                pipeline=None,
            )

    from nicegui.testing import user_simulation

    async with user_simulation(root=root) as user:
        await user.open("/")
        await user.should_see(content="Pregunta 1 — VALID")
        await user.should_see(content="Pregunta 2 — NEEDS_ATTENTION")
        await user.should_see("Aprobar y guardar esta pregunta")
        await user.should_see("Alternativa 1 — id")
        await user.should_see("Opciones Avanzadas")

        # La vista por defecto de la columna derecha es «Imagen original».
        await user.should_see("Imagen original")
        await user.should_see("No hay documento de origen para mostrar.")

        # Al cambiar a «Figuras nuevas» aparece la gestión de figuras.
        user.find("Figuras nuevas").click()
        await user.should_see("➕ Añadir nueva figura")


async def test_add_figure_area_renders_cropper_with_real_source(
    ui_env: None, tmp_path: Path
) -> None:
    """Regresión: con un origen real el alta de figuras dibuja el cropper.

    Antes, ``_render_add_area`` llamaba a ``Select.on_change`` (inexistente en
    NiceGUI 3), lo que abortaba la construcción del panel y solo dejaba ver la
    primera pregunta y un selector de página vacío.
    """
    from nicegui.testing import user_simulation
    from PIL import Image

    settings = get_settings()
    database = Database(settings=settings)
    staged = _make_questions()[:1]
    source = tmp_path / "pagina.png"
    Image.new("RGB", (800, 1000), "white").save(source)

    def root() -> None:
        session = SessionState()
        session.questions = {0: QuestionState()}
        QuestionPanel(
            ui.column().classes("w-full"),
            index=0,
            staged=staged[0],
            session=session,
            source_path=str(source),
            storage=_StubStorage(),
            settings=settings,
            database=database,
            pipeline=None,
        )

    async with user_simulation(root=root) as user:
        await user.open("/")
        await user.should_see("Pregunta 1 — VALID")

        # La numeración de páginas mostrada empieza en 1.
        await user.should_see("Página original 1")
        await user.should_not_see("Página original 0")

        # Cambiar a la pestaña de figuras: aquí se construía el alta de figuras.
        user.find("Figuras nuevas").click()
        await user.should_see("➕ Añadir nueva figura")

        # Desplegar el alta dibuja el cropper (render diferido).
        add_expansion = next(
            element
            for element in user.find("➕ Añadir nueva figura").elements
            if isinstance(element, ui.expansion)
        )
        with user:
            add_expansion.value = True
        await user.should_see("Arrastra y suelta para seleccionar la figura")
        await user.should_see("Añadir figura")


async def test_add_figure_keeps_cropper_selection(ui_env: None, tmp_path: Path) -> None:
    """Regresión: el recuadro se conserva al soltar y la figura se añade.

    Antes, ``DragBoxCropper`` borraba el rectángulo en ``mouseup`` y el
    ``on_confirm`` guardaba ``left/top/width/height`` en lugar de ``bbox``, de
    modo que el alta siempre avisaba de que no había selección.
    """
    from nicegui.testing import user_simulation
    from PIL import Image

    settings = get_settings()
    database = Database(settings=settings)
    staged = _make_questions()[1]  # sin figuras: un único cropper en el panel
    source = tmp_path / "pagina.png"
    Image.new("RGB", (600, 800), "white").save(source)
    storage = _FakeStorage()

    def root() -> None:
        session = SessionState()
        session.questions = {0: QuestionState()}
        QuestionPanel(
            ui.column().classes("w-full"),
            index=0,
            staged=staged,
            session=session,
            source_path=str(source),
            storage=storage,
            settings=settings,
            database=database,
            pipeline=None,
        )

    async with user_simulation(root=root) as user:
        await user.open("/")
        await user.should_see("Pregunta 1 — NEEDS_ATTENTION")

        user.find("Figuras nuevas").click()
        await user.should_see("➕ Añadir nueva figura")
        add_expansion = next(
            element
            for element in user.find("➕ Añadir nueva figura").elements
            if isinstance(element, ui.expansion)
        )
        with user:
            add_expansion.value = True

        cropper_ui = user.find(ui.interactive_image)
        cropper = next(
            element for element in cropper_ui.elements if isinstance(element, ui.interactive_image)
        )

        def mouse(kind: str, x: int, y: int) -> None:
            cropper_ui.trigger("mouse", {"mouse_event_type": kind, "image_x": x, "image_y": y})

        mouse("mousedown", 50, 60)
        mouse("mousemove", 250, 260)
        mouse("mouseup", 250, 260)
        # El recuadro sigue dibujado tras soltar el ratón.
        assert "<rect" in cropper.content

        user.find("Añadir figura").click()
        await user.should_see("fig:1 — recorte (página 1)", retries=15)
