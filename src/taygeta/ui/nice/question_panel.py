"""Panel de revisión de una pregunta en la UI NiceGUI.

Modelo de estado por eventos: la página se construye una vez por sesión y los
eventos mutan el estado (``QuestionState``) y refrescan los contenedores, sin
reruns.

El panel contiene:

* editor Markdown/TeX con previsualización enlazada (``ui.markdown``);
* tipo de pregunta y editor de alternativas (solo ``multiple_choice``);
* taxonomía con comboboxes nativos de curso/tema (``ui.select`` con
  ``new_value_mode="add-unique"``: elegir del catálogo o escribir un slug
  nuevo), subtema, dificultad y prerrequisitos;
* opciones avanzadas: contexto correctivo y reanálisis del enunciado (``refine``)
  que re-escribe los controles del panel con la respuesta del LLM;
* figuras a la derecha con pestañas «Imagen original» / «Figuras nuevas»,
  borrado, recorte manual y alta de falsos negativos con el cropper de arrastre;
  la numeración de páginas mostrada empieza en 1 (el índice interno es 0);
* checkbox de aprobación que bloquea la persistencia si el enunciado deja
  placeholders colgantes.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

from nicegui import run, ui
from PIL import Image

from taygeta.core.schema import (
    Choice,
    EnrichedQuestion,
    TranscriptionIntermediate,
)
from taygeta.core.settings import Settings
from taygeta.ingest.pipeline import IngestPipeline, StagedQuestion
from taygeta.ingest.validator import STATUS_NEEDS_ATTENTION, STATUS_VALID
from taygeta.server.db import Database
from taygeta.server.object_storage import ObjectStorage
from taygeta.ui.figure_editing import (
    bbox_to_pixel_box,
    pixel_box_to_normalized_bbox,
    reconcile_figures_with_statement,
    remove_figure,
)
from taygeta.ui.nice import caption
from taygeta.ui.nice.figures import (
    DragBoxCropper,
    crop_new_figure,
    figure_pil,
    page_pil,
    replace_figure_crop,
)
from taygeta.ui.nice.state import QuestionState, SessionState
from taygeta.ui.shared import (
    cache_busted_public_url,
    document_page_count,
    page_image_bytes,
    slug_split,
)

QUESTION_TYPES = ["multiple_choice", "open_ended", "true_false", "numerical"]


def _taxonomy_defaults(staged: StagedQuestion) -> dict[str, Any]:
    """Valores por defecto seguros (vía ``getattr``) para la taxonomía.

    El objeto en staging es un :class:`EnrichedQuestion` (el estructurador infiere
    la taxonomía en el pase único), por lo que ``course``, ``topic``, ``subtopic``,
    ``difficulty`` y ``prerequisites`` ya vienen rellenos. Se accede con
    ``getattr`` por robustez ante preguntas no estructuradas (``question=None``).
    """
    base = staged.question
    course = getattr(base, "course", "") or ""
    topic = getattr(base, "topic", "") or ""
    subtopic = getattr(base, "subtopic", None)
    difficulty = getattr(base, "difficulty", 3)
    prerequisites = getattr(base, "prerequisites", None) or []

    if not isinstance(subtopic, str) or not subtopic:
        subtopic = None
    if not isinstance(difficulty, int) or isinstance(difficulty, bool) or not 1 <= difficulty <= 5:
        difficulty = 3
    return {
        "course": course,
        "topic": topic,
        "subtopic": subtopic,
        "difficulty": difficulty,
        "prerequisites": list(prerequisites),
    }


def _register_orphan(
    session: SessionState, question_id: str | None, asset_path: str | None
) -> None:
    """Registra un asset de figura como huérfano para purgarlo al confirmar."""
    if not asset_path:
        return
    session.orphans.setdefault(str(question_id or ""), set()).add(asset_path)


@dataclass(slots=True)
class PanelDraft:
    """Recogida de los valores del panel en el momento de confirmar."""

    base: EnrichedQuestion
    statement: str
    qtype: str
    choices: list[Choice] | None
    course: str
    topic: str
    subtopic: str | None
    difficulty: int
    prerequisites: list[str]
    new_prereqs: list[str]
    effective_figures: list[dict[str, Any]]
    colgantes: list[str]
    approved: bool


class QuestionPanel:
    """Panel de revisión de una pregunta (editor, taxonomía, figuras, aprobación)."""

    def __init__(
        self,
        parent: ui.element,
        *,
        index: int,
        staged: StagedQuestion,
        session: SessionState,
        source_path: str | None,
        storage: ObjectStorage | None,
        settings: Settings,
        database: Database,
        pipeline: IngestPipeline,
    ) -> None:
        self.index = index
        self.staged = staged
        self.base = staged.question
        self.session = session
        self.source_path = source_path
        self.storage = storage
        self.settings = settings
        self.database = database
        self.pipeline = pipeline
        self.qstate = session.questions.get(index) or QuestionState()
        if self.base is not None:
            self.qstate.figures = [
                figure if isinstance(figure, dict) else figure.model_dump(mode="json")
                for figure in (self.base.figures or [])
            ]
            session.questions[index] = self.qstate
        self.view = "Página original"
        self._colgantes: list[str] = []

        # Referencias a los widgets (para confirmar y para reescribir tras un reanálisis).
        self.statement_area: ui.textarea | None = None
        self.preview: ui.markdown | None = None
        self.qtype_select: ui.select | None = None
        self.choices_area: ui.column | None = None
        self.choice_editors: list[tuple[ui.input, ui.textarea]] = []
        self.course_select: ui.select | None = None
        self.topic_select: ui.select | None = None
        self.subtopic_input: ui.input | None = None
        self.difficulty_slider: ui.slider | None = None
        self.prereq_select: ui.select | None = None
        self.hint_input: ui.input | None = None
        self.new_prereq_input: ui.input | None = None
        self.approve_box: ui.checkbox | None = None
        self.colgantes_label: ui.label | None = None
        self._refine_error: ui.label | None = None
        self._refine_spinner: ui.spinner | None = None
        self.right_body: ui.column | None = None
        self.figures_area: ui.column | None = None

        self._build(parent)

    # ------------------------------------------------------------------
    # Construcción del panel
    # ------------------------------------------------------------------

    def _build(self, parent: ui.element) -> None:
        staged = self.staged
        base = self.base
        with (
            parent,
            ui.expansion(
                f"Pregunta {self.index + 1} — {staged.status}",
                value=staged.status == STATUS_NEEDS_ATTENTION,
            ),
        ):
            if staged.status == STATUS_VALID:
                ui.label(
                    "Validación superada (VALID). Revísala y apruébala para guardarla."
                ).classes("text-emerald-700")
            else:
                ui.label("NEEDS_ATTENTION: " + "; ".join(staged.reasons)).classes("text-red-600")
            if staged.logical is not None and not staged.logical.is_valid:
                ui.label(f"Motivo de rechazo semántico: {staged.logical.rejection_reason}").classes(
                    "text-red-600"
                )
            if base is None:
                ui.label("No hay enunciado estructurado para editar en esta pregunta.")
                return
            # Sistema de dos columnas: a la izquierda el texto reconocido, la
            # vista formateada y los metadatos; a la derecha las figuras y la
            # página original en pestañas. Se usa un grid (no un row) porque
            # ``ui.row`` es ``flex-wrap: wrap`` y las columnas con gap se
            # apilarían al superar el 100% de ancho.
            with ui.grid().classes("w-full gap-4 grid-cols-1 lg:grid-cols-5 items-start"):
                with ui.column().classes("col-span-1 lg:col-span-3 gap-3 min-w-0"):
                    self._build_editor()
                    self._build_taxonomy()
                    self._build_advanced()
                    self.colgantes_label = ui.label("").classes("text-red-600 text-sm")
                    self.colgantes_label.set_visibility(False)
                    self.approve_box = ui.checkbox(
                        "Aprobar y guardar esta pregunta", value=staged.status == STATUS_VALID
                    )
                    self._sync_status()
                with ui.column().classes("col-span-1 lg:col-span-2 gap-3 min-w-0"):
                    with ui.tabs(
                        value="Página original",
                        on_change=lambda e: self._set_view(e.value),
                    ).classes("w-full"):
                        ui.tab("Página original", label="Imagen original")
                        ui.tab("Figuras", label="Figuras nuevas")
                    self.right_body = ui.column().classes("w-full gap-2")
                    self._set_view("Página original")

    def _build_editor(self) -> None:
        base = self.base
        if base is None:
            return
        self.statement_area = ui.textarea(
            "Enunciado (Markdown/TeX)", value=base.statement_raw
        ).classes("w-full")
        self.statement_area.props("outlined autogrow")
        self.preview = ui.markdown(base.statement_raw).classes("w-full border rounded p-2")
        self.statement_area.on_value_change(self._on_statement_change)

        self.qtype_select = ui.select(
            QUESTION_TYPES,
            label="Tipo de pregunta",
            value=base.question_type,
            with_input=True,
        )
        self.qtype_select.on_value_change(lambda _e: self._rebuild_choices())
        self.choices_area = ui.column().classes("w-full gap-2")
        self._rebuild_choices()
        if base.notation_notes:
            ui.label("Notas de notación: " + "; ".join(base.notation_notes)).classes(
                "text-sm text-gray-600"
            )
        ui.separator()
        ui.label("Metadatos").classes("font-semibold")

    def _rebuild_choices(self) -> None:
        """(Re)construye los editores de alternativas según el tipo seleccionado."""
        if self.choices_area is None or self.qtype_select is None:
            return
        self.choices_area.clear()
        self.choice_editors = []
        if self.qtype_select.value != "multiple_choice":
            return
        base_choices = self.base.choices if self.base is not None else []
        with self.choices_area:
            for choice_index, choice in enumerate(base_choices):
                with ui.row().classes("w-full gap-2 items-start"):
                    choice_id = ui.input(
                        f"Alternativa {choice_index + 1} — id", value=choice.id
                    ).classes("w-32")
                    choice_text = ui.textarea(
                        f"Alternativa {choice_index + 1} — contenido",
                        value=choice.content_raw,
                    ).classes("flex-1")
                self.choice_editors.append((choice_id, choice_text))

    def _combobox(self, label: str, options: list[str], value: str) -> ui.select:
        """Combobox de slug: catálogo desplegable + escritura de un valor nuevo.

        El valor actual se incluye siempre en las opciones para no romper cuando
        el catálogo aún está vacío o el valor inferido no está catalogado.
        """
        merged = sorted({*options, value}) if value else list(options)
        return ui.select(
            merged,
            label=label,
            value=value or None,
            with_input=True,
            new_value_mode="add-unique",
        )

    def _topics_for(self, course: str) -> list[str]:
        if not course:
            return []
        return sorted({row["topic_slug"] for row in self.database.list_topics(course)})

    def _build_taxonomy(self) -> None:
        defaults = _taxonomy_defaults(self.staged)
        courses = sorted({row["slug"] for row in self.database.list_courses()})
        self.course_select = self._combobox("Curso", courses, defaults["course"]).tooltip(
            "Curso del catálogo o escribe uno nuevo manualmente."
        )
        self.topic_select = self._combobox(
            "Tema", self._topics_for(defaults["course"]), defaults["topic"]
        ).tooltip("Temas del curso seleccionado o escribe uno nuevo manualmente.")
        self.course_select.on_value_change(self._on_course_change)

        self.subtopic_input = ui.input(
            "Subtema (slug snake_case, opcional)", value=defaults["subtopic"] or ""
        )
        with ui.row().classes("items-center gap-4"):
            self.difficulty_slider = ui.slider(
                min=1, max=5, step=1, value=int(defaults["difficulty"])
            )
            ui.label().bind_text_from(
                self.difficulty_slider, "value", backward=lambda v: f"Dificultad: {v}"
            )
        catalog = sorted({row["slug"] for row in self.database.list_prerequisites()})
        prereq_default = list(dict.fromkeys(defaults["prerequisites"]))
        options = sorted({*catalog, *prereq_default})
        self.prereq_select = ui.select(
            options,
            label="Prerrequisitos",
            value=prereq_default,
            multiple=True,
            with_input=True,
            new_value_mode="add-unique",
        )

    def _build_advanced(self) -> None:
        with ui.expansion("Opciones Avanzadas"):
            self.hint_input = ui.input(
                "Contexto correctivo (p. ej. «La condición ilegible es g(x) = ...»)"
            ).classes("w-full")
            self._refine_error = ui.label("").classes("text-red-600 text-sm")
            self._refine_error.set_visibility(False)
            with ui.row().classes("items-center gap-2"):
                ui.button("Reanalizar enunciado", on_click=self._refine)
                self._refine_spinner = ui.spinner(size="sm")
                self._refine_spinner.set_visibility(False)
            self.new_prereq_input = ui.input("Nuevos prerrequisitos (separados por coma)").classes(
                "w-full"
            )

    # ------------------------------------------------------------------
    # Eventos del editor y taxonomía
    # ------------------------------------------------------------------

    def _on_statement_change(self, e: Any) -> None:
        value = getattr(e, "value", None)
        if value is None and self.statement_area is not None:
            value = self.statement_area.value
        if self.preview is not None:
            self.preview.set_content(value or "")
        self._sync_status()

    def _on_course_change(self, e: Any) -> None:
        course = e.value or ""
        topics = self._topics_for(course)
        if self.topic_select is not None:
            self.topic_select.options = topics
            if self.topic_select.value not in topics:
                self.topic_select.value = None
            self.topic_select.update()

    def _sync_status(self) -> None:
        """Reconcilia figuras con el enunciado y actualiza el aviso de colgantes."""
        if self.statement_area is None:
            return
        statement = self.statement_area.value or ""
        _efectivas, huerfanas, colgantes = reconcile_figures_with_statement(
            self.qstate.figures, statement
        )
        for figure in huerfanas:
            _register_orphan(
                self.session, str(self.base.id) if self.base else None, figure.get("asset_path")
            )
        self._colgantes = colgantes
        if self.colgantes_label is None:
            return
        if colgantes:
            self.colgantes_label.set_text(
                "El enunciado referencia placeholders sin figura declarada: "
                + ", ".join(sorted(colgantes))
                + ". Elimina el placeholder o restaura la figura para poder aprobar."
            )
            self.colgantes_label.set_visibility(True)
        else:
            self.colgantes_label.set_visibility(False)

    # ------------------------------------------------------------------
    # Figuras (columna derecha)
    # ------------------------------------------------------------------

    def _set_view(self, view: str) -> None:
        self.view = view
        if self.right_body is None:
            return
        self.right_body.clear()
        # Al salir de «Figuras» invalidamos la referencia para que las
        # actualizaciones (refine, borrado) no intenten redibujar sobre una
        # vista ya destruida; se reconstruye al volver a la pestaña.
        self.figures_area = None
        with self.right_body:
            if view == "Figuras":
                self.figures_area = ui.column().classes("w-full gap-2")
                self._render_figures()
            else:
                self._render_original_pages()

    def _render_figures(self) -> None:
        """Dibuja (o redibuja) la lista de figuras y el alta manual."""
        if self.figures_area is None:
            return
        self.figures_area.clear()
        with self.figures_area:
            ui.label("Figuras de la pregunta").classes("font-semibold")
            for figure in self.qstate.figures:
                figure_id = figure.get("id") or "fig:?"
                page_index = int(figure.get("page", 0))
                with ui.card().classes("w-full gap-1"):
                    image = figure_pil(self.storage, figure.get("asset_path"))
                    if image is not None:
                        ui.image(image).classes("w-full rounded border")
                    else:
                        caption(
                            f"{figure_id}: no se pudo cargar el recorte (página {page_index + 1})"
                        )
                    caption(f"{figure_id} — recorte (página {page_index + 1})")
                    url = cache_busted_public_url(
                        figure.get("asset_path"), self.settings, seed=figure.get("bbox")
                    )
                    if url:
                        ui.link(f"Abrir {figure_id} en el navegador", url, new_tab=True).classes(
                            "text-xs"
                        )
                    ui.button(
                        "Eliminar figura",
                        on_click=lambda fid=figure_id: self._delete_figure(fid),
                    ).props("flat color=negative dense")
                    with ui.expansion(f"Recortar manualmente — {figure_id}"):
                        self._render_recrop(figure_id, figure, page_index)
            self._render_add_area()

    def _render_recrop(self, figure_id: str, figure: dict[str, Any], page_index: int) -> None:
        result = page_pil(self.source_path, page_index)
        if result is None:
            caption("No hay documento de origen para recortar esta figura.")
            return
        image, width, height = result
        pending: dict[str, Any] = {"bbox": None}
        DragBoxCropper(
            image,
            width=width,
            height=height,
            on_confirm=lambda box: pending.update({"bbox": box}),
            caption=f"Arrastra sobre la página {page_index + 1} para ajustar el recorte",
            initial_box=self._initial_pixel_box(figure, width, height),
        )
        ui.button(
            "Aplicar recorte",
            on_click=self._apply_recrop(figure_id, width, height, pending),
        ).props("dense")

    @staticmethod
    def _initial_pixel_box(
        figure: dict[str, Any], width: int, height: int
    ) -> dict[str, int] | None:
        """Caja en píxeles de la BBox actual de la figura, para sembrar el cropper.

        Devuelve el recuadro vigente ``{"left", "top", "width", "height"}`` o
        ``None`` si la figura no tiene BBox válida para las dimensiones dadas.
        """
        bbox = figure.get("bbox")
        if not bbox:
            return None
        try:
            xl, xr, yt, yb = bbox_to_pixel_box(bbox, int(width), int(height))
        except (ValueError, TypeError):
            return None
        return {"left": xl, "top": yt, "width": xr - xl, "height": yb - yt}

    def _apply_recrop(
        self, figure_id: str, width: int, height: int, pending: dict[str, Any]
    ) -> Any:
        async def handler() -> None:
            box = pending.get("bbox")
            if box is None:
                ui.notify("Arrastra sobre la página para seleccionar la región", type="warning")
                return
            try:
                new_bbox = pixel_box_to_normalized_bbox(box, int(width), int(height))
                figures = await run.io_bound(
                    replace_figure_crop,
                    self.qstate.figures,
                    figure_id,
                    new_bbox,
                    self.source_path,
                    self.storage,
                )
            except Exception as exc:  # noqa: BLE001 - se informa al usuario, no se oculta
                ui.notify(f"No se pudo aplicar el recorte de {figure_id}: {exc}", type="negative")
                return
            self.qstate.figures = figures
            ui.notify(f"{figure_id} recortada y guardada.", type="positive")
            self._render_figures()

        return handler

    def _render_add_area(self) -> None:
        with ui.expansion("➕ Añadir nueva figura") as add_exp:
            page_count = document_page_count(self.source_path)
            if page_count is None or page_count <= 0:
                caption("No hay documento de origen para añadir figuras.")
                return
            default_page = 0
            if self.qstate.figures:
                default_page = min(int(self.qstate.figures[0].get("page", 0)), int(page_count) - 1)
            page_select = ui.select(
                {index: f"Página {index + 1}" for index in range(int(page_count))},
                value=default_page,
                label="Página original del documento",
            )
            body = ui.column().classes("w-full gap-2")
            pending: dict[str, Any] = {"bbox": None, "page": int(default_page)}
            dims: dict[str, int] = {}

            def show_cropper() -> None:
                body.clear()
                with body:
                    page_index = int(page_select.value)
                    result = page_pil(self.source_path, page_index)
                    if result is None:
                        caption("No se pudo cargar la página original para recortar.")
                        return
                    image, width, height = result
                    dims["width"], dims["height"] = width, height
                    pending.clear()
                    pending["page"] = page_index
                    pending["bbox"] = None
                    DragBoxCropper(
                        image,
                        width=width,
                        height=height,
                        on_confirm=lambda box: pending.update({"bbox": box}),
                        caption="Arrastra y suelta para seleccionar la figura",
                    )
                    ui.button("Añadir figura", on_click=self._add_figure(pending, dims)).props(
                        "dense"
                    )

            page_select.on_value_change(lambda _e: show_cropper())
            # El cropper se dibuja al abrir la expansión (no en build time) para
            # no rasterizar una página por cada panel al construir la revisión.
            add_exp.on_value_change(lambda e: show_cropper() if e.value else None)

    def _add_figure(self, pending: dict[str, Any], dims: dict[str, int]) -> Any:
        async def handler() -> None:
            box = pending.get("bbox")
            if box is None or self.statement_area is None:
                ui.notify("Arrastra sobre la página para seleccionar la región", type="warning")
                return
            try:
                new_bbox = pixel_box_to_normalized_bbox(box, dims["width"], dims["height"])
                figures, statement, figure_id = await run.io_bound(
                    crop_new_figure,
                    self.qstate.figures,
                    self.statement_area.value,
                    page_index=int(pending["page"]),
                    bbox=list(new_bbox),
                    source_path=self.source_path,
                    storage=self.storage,
                )
            except Exception as exc:  # noqa: BLE001 - se informa al usuario, no se oculta
                ui.notify(f"No se pudo añadir la figura: {exc}", type="negative")
                return
            self.qstate.figures = figures
            self.statement_area.value = statement
            if self.preview is not None:
                self.preview.set_content(statement)
            self._sync_status()
            ui.notify(
                f"{figure_id} añadida; su placeholder quedó al final del enunciado.",
                type="positive",
            )
            self._render_figures()

        return handler

    def _delete_figure(self, figure_id: str) -> None:
        if self.statement_area is None:
            return
        try:
            figures, statement, removed = remove_figure(
                self.qstate.figures, self.statement_area.value, figure_id
            )
        except Exception as exc:  # noqa: BLE001 - se informa al usuario, no se oculta
            ui.notify(f"No se pudo eliminar la figura: {exc}", type="negative")
            return
        self.qstate.figures = figures
        for figure in removed:
            _register_orphan(
                self.session,
                str(self.base.id) if self.base else None,
                figure.get("asset_path"),
            )
        self.statement_area.value = statement
        if self.preview is not None:
            self.preview.set_content(statement)
        self._sync_status()
        self._render_figures()

    def _render_original_pages(self) -> None:
        if self.source_path is None:
            caption("No hay documento de origen para mostrar.")
            return
        pages = [0]
        if self.base is not None and self.base.figures:
            pages = sorted({figure.page for figure in self.base.figures})
        for page_index in pages:
            data = page_image_bytes(self.source_path, page_index)
            if data:
                ui.image(Image.open(io.BytesIO(data))).classes("w-full rounded border")
                caption(f"Página original {page_index + 1}")
            else:
                caption(f"No se pudo rasterizar la página {page_index + 1}.")

    # ------------------------------------------------------------------
    # Reanálisis del enunciado (refine)
    # ------------------------------------------------------------------

    async def _refine(self) -> None:
        result = self.session.staging
        if result is None or self.base is None:
            return
        hint = self.hint_input.value or None if self.hint_input is not None else None
        intermediate = result.intermediate
        if intermediate is None:
            intermediate = TranscriptionIntermediate(
                source_reference=result.source_reference,
                raw_markdown=self.base.statement_raw,
                pages=[],
                figures=self.base.figures,
                warnings=[],
            )
        if self._refine_error is not None:
            self._refine_error.set_visibility(False)
        if self._refine_spinner is not None:
            self._refine_spinner.set_visibility(True)
        try:
            refined = await run.io_bound(
                self.pipeline.refine_question, intermediate, self.base, hints=hint
            )
        except Exception as exc:  # noqa: BLE001 - se informa al panel, no se oculta
            if self._refine_error is not None:
                self._refine_error.set_text(f"No se pudo reanalizar: {exc}")
                self._refine_error.set_visibility(True)
            return
        finally:
            if self._refine_spinner is not None:
                self._refine_spinner.set_visibility(False)
        result.questions[self.index] = refined
        self._apply_refine(refined)

    def _apply_refine(self, staged: StagedQuestion) -> None:
        """Reescribe los controles del panel con el resultado del reanálisis."""
        question = staged.question
        if question is None:
            return
        self.base = question
        self.qstate.figures = [
            figure if isinstance(figure, dict) else figure.model_dump(mode="json")
            for figure in (question.figures or [])
        ]
        if self.statement_area is not None:
            self.statement_area.value = question.statement_raw
        if self.preview is not None:
            self.preview.set_content(question.statement_raw)
        if self.qtype_select is not None:
            self.qtype_select.value = question.question_type
        self._rebuild_choices()
        if self.course_select is not None:
            self.course_select.value = question.course
        if self.topic_select is not None:
            topics = self._topics_for(question.course)
            self.topic_select.options = topics
            self.topic_select.value = question.topic or None
            self.topic_select.update()
        if self.subtopic_input is not None:
            self.subtopic_input.value = question.subtopic or ""
        if self.difficulty_slider is not None:
            self.difficulty_slider.value = int(question.difficulty)
        if self.prereq_select is not None:
            options = sorted({*self.prereq_select.options, *question.prerequisites})
            self.prereq_select.options = options
            self.prereq_select.value = list(dict.fromkeys(question.prerequisites))
            self.prereq_select.update()
        self._sync_status()
        self._render_figures()

    # ------------------------------------------------------------------
    # Confirmación
    # ------------------------------------------------------------------

    def collect(self) -> PanelDraft | None:
        """Recoge los valores actuales del panel (o ``None`` si no hay enunciado)."""
        base = self.base
        if base is None or self.statement_area is None or self.qtype_select is None:
            return None
        statement = self.statement_area.value or ""
        qtype = str(self.qtype_select.value or base.question_type)
        choices: list[Choice] | None = None
        if qtype == "multiple_choice":
            choices = [
                Choice(id=choice_id.value.strip(), content_raw=choice_text.value.strip())
                for choice_id, choice_text in self.choice_editors
            ]
        effective, huerfanas, colgantes = reconcile_figures_with_statement(
            self.qstate.figures, statement
        )
        for figure in huerfanas:
            _register_orphan(self.session, str(base.id), figure.get("asset_path"))
        course = str(self.course_select.value or "") if self.course_select is not None else ""
        topic = str(self.topic_select.value or "") if self.topic_select is not None else ""
        subtopic = self.subtopic_input.value if self.subtopic_input is not None else ""
        difficulty = (
            int(self.difficulty_slider.value)
            if self.difficulty_slider is not None and self.difficulty_slider.value
            else 3
        )
        prerequisites = (
            list(self.prereq_select.value or []) if self.prereq_select is not None else []
        )
        new_prereqs = (
            slug_split(self.new_prereq_input.value) if self.new_prereq_input is not None else []
        )
        approved = bool(self.approve_box.value) if self.approve_box is not None else False
        return PanelDraft(
            base=base,
            statement=statement,
            qtype=qtype,
            choices=choices,
            course=course,
            topic=topic,
            subtopic=slug_split(subtopic)[0] if subtopic else None,
            difficulty=difficulty,
            prerequisites=prerequisites,
            new_prereqs=new_prereqs,
            effective_figures=effective,
            colgantes=colgantes,
            approved=approved,
        )
