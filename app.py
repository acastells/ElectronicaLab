from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import json
import sys
import threading
import traceback
from functools import partial
from pathlib import Path
from urllib.parse import urlencode

from PySide6.QtCore import Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QActionGroup, QDesktopServices
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStyle,
    QStyleOptionSlider,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core import (
    BASE_VERSION,
    CLAP_VERSION,
    ClapEncoder,
    Library,
    analyze,
    content_hash,
    data_directory,
    predict_personal,
    train_personal,
)
from overview import decode_overview
from timeline import AudioTimeline


class ClickSlider(QSlider):
    """Move immediately to the clicked position, retaining native dragging."""

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        self.setSliderDown(True)
        self.move_to_pointer(event)

    def keyPressEvent(self, event):
        self.setSliderDown(True)
        super().keyPressEvent(event)
        self.setSliderDown(False)

    def mouseMoveEvent(self, event):
        if self.isSliderDown():
            self.move_to_pointer(event)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.isSliderDown():
            self.move_to_pointer(event)
            self.setSliderDown(False)
        else:
            super().mouseReleaseEvent(event)

    def move_to_pointer(self, event):
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        handle = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderHandle,
            self,
        )
        span = max(1, self.width() - handle.width())
        position = round(event.position().x() - handle.width() / 2)
        value = QStyle.sliderValueFromPosition(
            self.minimum(), self.maximum(), position, span, option.upsideDown
        )
        self.setValue(value)
        event.accept()


class SortableItem(QTableWidgetItem):
    def __init__(self, text, sort_value):
        super().__init__(text)
        self.sort_value = sort_value

    def __lt__(self, other):
        return self.sort_value < other.sort_value


class ChartWorker(QThread):
    ready = Signal(object)
    failed = Signal(str)

    def __init__(self, request):
        super().__init__()
        self.request = request
        self.stop = threading.Event()

    def run(self):
        try:
            result = dict(self.request)
            overview = decode_overview(result["path"], result["duration"], self.stop)
            if overview is not None:
                result["overview"] = overview
                self.ready.emit(result)
        except Exception as error:
            self.failed.emit(str(error))


class Worker(QThread):
    message = Signal(str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, root, action, argument=None, engine="acoustic"):
        super().__init__()
        self.root = root
        self.action = action
        self.argument = argument
        self.engine = engine
        self.stop = threading.Event()

    def run(self):
        lib = None
        try:
            lib = Library(self.root)
            if self.action == "scan":
                n = lib.scan(self.argument, self.message.emit, self.stop)
                result = f"{n} archivos importados o actualizados."
            elif self.action == "analyze":
                result = self.analyze_library(lib)
            elif self.action == "train":
                report = train_personal(
                    lib, CLAP_VERSION if self.engine == "clap" else BASE_VERSION
                )
                count = predict_personal(lib)
                result = (
                    f"Modelo actualizado; {count} canciones con sugerencias.\n"
                    + json.dumps(report, ensure_ascii=False, indent=2)
                )
            elif self.action == "predict":
                result = f"Sugerencias actualizadas: {predict_personal(lib)} canciones."
            else:
                raise ValueError("Acción desconocida")
            self.done.emit(result)
        except Exception as e:
            self.failed.emit(str(e) + "\n" + traceback.format_exc())
        finally:
            if lib:
                lib.close()

    def analyze_library(self, lib):
        version = CLAP_VERSION if self.engine == "clap" else BASE_VERSION
        encoder = ClapEncoder(self.root) if self.engine == "clap" else None
        rows = [
            r
            for r in lib.rows()
            if r["status"] != "done" or r["feature_version"] != version
        ]
        errors = 0
        processed = 0
        for index, row in enumerate(rows):
            if self.stop.is_set():
                break
            self.message.emit(f"{index + 1}/{len(rows)} · {row['title']}")
            try:
                stat = Path(row["path"]).stat()
                if stat.st_size != row["size"] or stat.st_mtime_ns != row["mtime"]:
                    raise ValueError(
                        "Archivo modificado: vuelve a importar la carpeta antes de analizar."
                    )
                if not row["duration"]:
                    raise ValueError("No se pudo leer la duración de este formato.")
                analysis, features = analyze(
                    row["path"],
                    row["duration"],
                    row["sr"],
                    self.engine,
                    encoder,
                )
                fingerprint = content_hash(row["path"])
                lib.update(
                    row["id"],
                    analysis=json.dumps(analysis),
                    features=json.dumps(features),
                    feature_version=version,
                    content_hash=fingerprint,
                    status="done",
                    error=None,
                    predictions="[]",
                )
                processed += 1
            except Exception as e:
                errors += 1
                lib.update(row["id"], status="error", error=str(e))
                self.message.emit(f"Error · {row['title']}: {e}")
        return f"Analizadas: {processed}. Errores: {errors}." + (
            " Pausado; pulsa Analizar para continuar." if self.stop.is_set() else ""
        )


class Window(QMainWindow):
    def __init__(self, root=None):
        super().__init__()
        self.root = Path(root) if root else data_directory()
        self.lib = Library(self.root)
        self.chart_worker = None
        self.chart_request = None
        self.chart_pending = None
        self.chart_timer = QTimer(self)
        self.chart_timer.setSingleShot(True)
        self.chart_timer.setInterval(150)
        self.chart_timer.timeout.connect(self.load_chart)
        self.worker = None
        self.current = None
        self.loaded = None
        self.cache = {}
        self.refreshing = False
        self.setWindowTitle("ElectronicaLab · Biblioteca y clasificación")
        self.resize(1440, 920)
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.audio.setVolume(0.65)
        host = QWidget()
        self.setCentralWidget(host)
        outer = QVBoxLayout(host)
        outer.setContentsMargins(20, 16, 20, 16)
        title = QLabel("ELECTRONICA LAB")
        title.setStyleSheet("font-size:25px;font-weight:700;color:#78e4c8")
        outer.addWidget(title)
        outer.addWidget(QLabel("Tu biblioteca · tus estilos · aprendizaje local"))
        self.build_menus()
        self.build_filters(outer)
        self.build_library_panel(outer)
        self.build_playback(outer)
        self.build_charts(outer)
        self.build_view_menu()
        self.apply_theme()
        self.reload_styles()
        self.refresh()
        self.table.setSortingEnabled(True)
        self.table.sortItems(0)
        self.restore_columns()
        self.log.append(
            "Empieza importando una carpeta. Las sugerencias requieren ejemplos revisados y entrenamiento. Los scores no son probabilidades calibradas."
        )

    def add_menu_action(self, menu, title, callback, busy=True):
        action = menu.addAction(title)
        action.triggered.connect(callback)
        if busy:
            self.busy_controls.append(action)
        return action

    def build_menus(self):
        self.busy_controls = []
        library_menu = self.menuBar().addMenu("Biblioteca")
        self.add_menu_action(library_menu, "Importar carpeta…", self.import_folder)
        library_menu.addSeparator()
        self.add_menu_action(library_menu, "Exportar CSV…", self.export_csv)
        self.add_menu_action(library_menu, "Copia de seguridad…", self.backup)
        self.add_menu_action(library_menu, "Abrir datos", self.open_data, busy=False)
        analysis_menu = self.menuBar().addMenu("Análisis")
        self.add_menu_action(
            analysis_menu, "Analizar / continuar", partial(self.start, "analyze")
        )
        self.pause = self.add_menu_action(
            analysis_menu, "Pausar tarea", self.stop_work, busy=False
        )
        self.pause.setEnabled(False)
        self.engine = analysis_menu.addMenu("Motor de análisis")
        self.engine_choices = QActionGroup(self)
        self.engine_choices.setExclusive(True)
        for title, name in [
            ("Ligero · características acústicas", "acoustic"),
            ("CLAP · modelo preentrenado", "clap"),
        ]:
            action = self.engine.addAction(title)
            action.setCheckable(True)
            action.setData(name)
            self.engine_choices.addAction(action)
            action.setChecked(name == "acoustic")
        model_menu = self.menuBar().addMenu("Clasificador")
        self.add_menu_action(
            model_menu, "Entrenar con mis etiquetas", partial(self.start, "train")
        )
        self.add_menu_action(
            model_menu, "Actualizar sugerencias", partial(self.start, "predict")
        )
        self.add_menu_action(model_menu, "Ver informe del modelo", self.model_report)

    def build_filters(self, outer):
        filters = QHBoxLayout()
        outer.addLayout(filters)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Buscar título, artista o ruta…")
        self.search.textChanged.connect(self.apply_filter)
        filters.addWidget(self.search, 2)
        self.style_filter = QComboBox()
        self.style_filter.currentTextChanged.connect(self.apply_filter)
        filters.addWidget(self.style_filter)
        self.state_filter = QComboBox()
        self.state_filter.addItems(
            [
                "Todos",
                "Sin revisar",
                "Revisados",
                "Pendientes de análisis",
                "Errores",
                "Corte detectado",
            ]
        )
        self.state_filter.currentTextChanged.connect(self.apply_filter)
        filters.addWidget(self.state_filter)
        columns = QPushButton("Columnas…")
        columns.clicked.connect(self.choose_columns)
        filters.addWidget(columns)

    def build_library_panel(self, outer):
        split = QSplitter()
        outer.addWidget(split, 1)
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            [
                "Título / artista",
                "Estilos confirmados",
                "Sugerencias · score",
                "BPM ≈",
                "kbps",
                "kHz",
                "Corte ≈",
                "Estado",
                "Ruta del archivo",
            ]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.table.setColumnWidth(0, 300)
        self.table.setColumnWidth(1, 170)
        self.table.setColumnWidth(2, 220)
        self.table.setColumnWidth(8, 280)
        for c in range(3, 8):
            self.table.setColumnWidth(c, 75)
        self.table.itemSelectionChanged.connect(self.select)
        self.table.itemDoubleClicked.connect(self.play)
        split.addWidget(self.table)
        self.build_labels_panel(split)

    def build_labels_panel(self, split):
        panel = QWidget()
        side = QVBoxLayout(panel)
        split.addWidget(panel)
        split.setSizes([980, 380])
        self.build_suggestions(side)
        side.addWidget(QLabel("TODOS LOS GÉNEROS"))
        side.addWidget(QLabel("Pulsa un género para añadirlo o quitarlo al instante."))
        self.genre_buttons = {}
        self.genre_host = QWidget()
        self.genre_layout = QGridLayout(self.genre_host)
        self.genre_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.genre_host)
        scroll.setMinimumHeight(170)
        side.addWidget(scroll, 1)
        self.busy_controls.append(scroll)
        self.styles = QComboBox()
        side.addWidget(self.styles)
        style_actions = QHBoxLayout()
        side.addLayout(style_actions)
        for text, fn in [
            ("+ Estilo", self.add_style),
            ("Renombrar", self.rename_style),
            ("Eliminar género", self.delete_style),
        ]:
            b = QPushButton(text)
            b.clicked.connect(fn)
            style_actions.addWidget(b)
            self.busy_controls.append(b)
        self.google_button = QPushButton("Buscar género en Google")
        self.google_button.setEnabled(False)
        self.google_button.clicked.connect(self.search_genre_online)
        side.addWidget(self.google_button)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(150)
        side.addWidget(self.details)
        self.stats = QLabel()
        self.stats.setWordWrap(True)
        side.addWidget(self.stats)

    def build_playback(self, outer):
        playback = QHBoxLayout()
        outer.addLayout(playback)
        b = QPushButton("▶ / Ⅱ")
        b.clicked.connect(self.play)
        playback.addWidget(b)
        self.now_playing = QLabel("Ahora sonando: ninguna canción")
        self.now_playing.setTextFormat(Qt.TextFormat.PlainText)
        self.now_playing.setWordWrap(True)
        playback.addWidget(self.now_playing, 1)
        self.player.playbackStateChanged.connect(self.update_now_playing)
        self.time = QLabel("0:00 / 0:00")
        playback.addWidget(self.time)
        self.player.durationChanged.connect(self.duration_changed)
        self.player.positionChanged.connect(self.position)
        self.player.errorOccurred.connect(self.playback_error)
        volume = ClickSlider(Qt.Orientation.Horizontal)
        volume.setRange(0, 100)
        volume.setValue(65)
        volume.setMaximumWidth(100)
        volume.valueChanged.connect(self.volume_changed)
        playback.addWidget(QLabel("Vol."))
        playback.addWidget(volume)
        delete_button = QPushButton("Eliminar archivo del dispositivo")
        delete_button.clicked.connect(self.delete_tracks)
        playback.addWidget(delete_button)
        self.busy_controls.append(delete_button)

    def build_charts(self, outer):
        self.timeline = AudioTimeline()
        self.timeline.seek_requested.connect(self.seek_changed)
        outer.addWidget(self.timeline)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(85)
        outer.addWidget(self.log)

    def build_view_menu(self):
        menu = self.menuBar().addMenu("Ver")
        self.panel_actions = {}
        panels = {
            "activity": ("Registro de actividad", [self.log]),
            "details": (
                "Detalles de la canción y estadísticas",
                [self.details, self.stats],
            ),
        }
        for key, (title, widgets) in panels.items():
            setting = self.lib.db.execute(
                "SELECT value FROM settings WHERE key=?", ("panel_" + key,)
            ).fetchone()
            visible = setting is not None and setting[0] == "1"
            action = menu.addAction(title)
            action.setCheckable(True)
            action.setChecked(visible)
            for widget in widgets:
                widget.setVisible(visible)
            action.toggled.connect(partial(self.set_panel_visible, key, widgets))
            self.panel_actions[key] = action

    def set_panel_visible(self, key, widgets, visible):
        for widget in widgets:
            widget.setVisible(visible)
        with self.lib.db:
            self.lib.db.execute(
                "INSERT OR REPLACE INTO settings VALUES (?, ?)",
                ("panel_" + key, "1" if visible else "0"),
            )

    def build_suggestions(self, layout):
        layout.addWidget(QLabel("SUGERENCIAS · TOP 3"))
        self.suggestion_hint = QLabel(
            "Selecciona una canción para ver sus sugerencias."
        )
        self.suggestion_hint.setWordWrap(True)
        layout.addWidget(self.suggestion_hint)
        self.suggestion_buttons = []
        for index in range(3):
            button = QPushButton()
            button.clicked.connect(partial(self.assign_suggestion, index))
            button.hide()
            layout.addWidget(button)
            self.suggestion_buttons.append(button)
        self.visible_suggestions = []

    def update_suggestions(self, ids):
        self.visible_suggestions = []
        if len(ids) == 1:
            row = self.cache[ids[0]]
            predictions = json.loads(row["predictions"])
            valid = [
                (name, score)
                for name, score in predictions
                if name in self.genre_buttons
            ]
            self.visible_suggestions = sorted(
                valid, key=lambda suggestion: suggestion[1], reverse=True
            )[:3]
        self.suggestion_hint.setVisible(not self.visible_suggestions)
        if len(ids) == 1:
            self.suggestion_hint.setText("Sin sugerencias guardadas para esta canción.")
        else:
            self.suggestion_hint.setText(
                "Selecciona una sola canción para ver sus sugerencias."
            )
        busy = self.worker is not None and self.worker.isRunning()
        for index, button in enumerate(self.suggestion_buttons):
            available = index < len(self.visible_suggestions)
            button.setVisible(available)
            button.setEnabled(available and not busy)
            if available:
                name, score = self.visible_suggestions[index]
                button.setText(f"{name} · {score:.2f}")
                button.setToolTip("Asignar este género y conservar los demás.")

    def assign_suggestion(self, index):
        ids = self.selected_ids()
        if len(ids) != 1 or index >= len(self.visible_suggestions):
            return
        if self.worker and self.worker.isRunning():
            return
        name, _ = self.visible_suggestions[index]
        labels = set(json.loads(self.lib.track(ids[0])["labels"]))
        labels.add(name)
        self.lib.set_labels(ids, sorted(labels))
        self.update_track_rows(ids)
        self.log.append(f"Género asignado: {name}.")

    def apply_theme(self):
        self.setStyleSheet(
            """QMenuBar {background:#172638;padding:4px;border-bottom:1px solid #36516b;}
            QMenuBar::item {padding:8px 14px;margin:0 3px;border-radius:5px;}
            QMenuBar::item:selected {background:#2c4860;color:#78e4c8;}
            QMenuBar::item:pressed {background:#235b67;color:#ffffff;}
            QMenu {background:#172638;border:1px solid #36516b;padding:6px;}
            QMenu::item {padding:8px 28px 8px 20px;border-radius:4px;}
            QMenu::item:selected {background:#235b67;color:#ffffff;}
            QMenu::item:disabled {color:#65788b;}
            QMenu::separator {height:1px;background:#36516b;margin:5px 8px;}
            QMainWindow,QWidget {background:#111d2b;color:#e5edf5;font-size:12px;} QPushButton {background:#20354a;border:1px solid #36516b;border-radius:5px;padding:8px;} QPushButton:hover {background:#2c4860;} QPushButton:disabled {color:#65788b;} QLineEdit,QComboBox,QListWidget,QTextEdit,QTableWidget {background:#172638;border:1px solid #2e4258;selection-background-color:#235b67;padding:4px;} QHeaderView::section {background:#22374b;padding:7px;border:0;} QSlider::groove:horizontal {height:5px;background:#355269;} QSlider::handle:horizontal {background:#78e4c8;width:12px;margin:-4px 0;border-radius:5px;}"""
        )

    def open_data(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.root)))

    def duration_changed(self, milliseconds):
        self.timeline.set_duration(int(milliseconds))

    def volume_changed(self, value):
        self.audio.setVolume(value / 100)

    def playback_error(self, *args):
        self.statusBar().showMessage("Reproducción: " + self.player.errorString())

    def reload_styles(self):
        old = self.style_filter.currentText()
        self.style_filter.blockSignals(True)
        self.style_filter.clear()
        self.style_filter.addItem("Todos los estilos")
        self.style_filter.addItems(self.lib.styles())
        self.style_filter.setCurrentText(old or "Todos los estilos")
        self.style_filter.blockSignals(False)
        self.styles.clear()
        self.styles.addItems(self.lib.styles())
        while self.genre_layout.count():
            item = self.genre_layout.takeAt(0)
            item.widget().deleteLater()
        self.genre_buttons.clear()
        for index, name in enumerate(self.lib.styles()):
            button = QPushButton(name)
            button.setCheckable(True)
            button.setStyleSheet(
                "QPushButton:checked {background:#235b67;border:1px solid #78e4c8;}"
            )
            button.clicked.connect(partial(self.toggle_genre, name))
            self.genre_layout.addWidget(button, index // 2, index % 2)
            self.genre_buttons[name] = button

    def refresh(self):
        selected = set(self.selected_ids()) if self.table.rowCount() else set()
        rows = self.lib.rows()
        self.refreshing = True
        self.table.setSortingEnabled(False)
        self.table.clearSelection()
        self.table.setRowCount(len(rows))
        self.cache = {r["id"]: r for r in rows}
        for i, r in enumerate(rows):
            self.populate_row(i, r)
            if r["id"] in selected:
                for c in range(self.table.columnCount()):
                    self.table.item(i, c).setSelected(True)
        self.table.setSortingEnabled(True)
        self.refreshing = False
        self.apply_filter()
        self.stats.setText(
            f"{len(rows)} canciones · {sum(r['reviewed'] for r in rows)} revisadas\nDatos: {self.root}"
        )
        self.select()

    def populate_row(self, i, r):
        a = json.loads(r["analysis"] or "{}")
        pred = json.loads(r["predictions"])
        labels = json.loads(r["labels"])
        values = [
            r["title"] + (" — " + r["artist"] if r["artist"] else ""),
            ", ".join(labels),
            " / ".join(f"{n}: {v:.2f}" for n, v in pred),
            str(a.get("bpm", "—")),
            f"{r['bitrate']:.0f}" if r["bitrate"] else "—",
            f"{r['sr'] / 1000:g}" if r["sr"] else "—",
            f"{a['cutoff_hz'] / 1000:g}" if a.get("cutoff_hz") else "—",
            {"done": "Analizado", "pending": "Pendiente", "error": "Error"}.get(
                r["status"], r["status"]
            ),
            r["path"],
        ]
        sort_values = [value.casefold() for value in values]
        sort_values[3:7] = [
            a.get("bpm") or -1,
            r["bitrate"] or -1,
            r["sr"] or -1,
            a.get("cutoff_hz") or -1,
        ]
        sort_values[2] = max((score for _, score in pred), default=-1)
        for c, value in enumerate(values):
            item = SortableItem(value, sort_values[c])
            item.setData(Qt.ItemDataRole.UserRole, r["id"])
            item.setToolTip(r["error"] or r["path"])
            self.table.setItem(i, c, item)

    def apply_filter(self, *args):
        query = self.search.text().casefold()
        style = self.style_filter.currentText()
        state = self.state_filter.currentText()
        for i in range(self.table.rowCount()):
            r = self.cache[self.table.item(i, 0).data(Qt.ItemDataRole.UserRole)]
            a = json.loads(r["analysis"] or "{}")
            visible = (
                query in (r["title"] + " " + r["artist"] + " " + r["path"]).casefold()
            )
            visible &= style in ("", "Todos los estilos") or style in json.loads(
                r["labels"]
            )
            visible &= {
                "Todos": True,
                "Sin revisar": not r["reviewed"],
                "Revisados": bool(r["reviewed"]),
                "Pendientes de análisis": r["status"] != "done",
                "Errores": r["status"] == "error",
                "Corte detectado": bool(a.get("cutoff_hz")),
            }.get(state, True)
            self.table.setRowHidden(i, not visible)

    def selected_ids(self):
        return list(
            dict.fromkeys(
                self.table.item(ix.row(), 0).data(Qt.ItemDataRole.UserRole)
                for ix in self.table.selectionModel().selectedRows()
                if not self.table.isRowHidden(ix.row())
            )
        )

    def select(self):
        if self.refreshing:
            return
        ids = self.selected_ids()
        self.google_button.setEnabled(len(ids) == 1)
        if not ids:
            self.current = None
            self.details.clear()
            self.update_genre_buttons([])
            return
        r = self.lib.track(ids[0])
        self.current = r["id"]
        self.update_genre_buttons(ids)
        a = json.loads(r["analysis"] or "{}")
        compact = {
            k: v for k, v in a.items() if k not in ("frequency_hz", "spectrum_db")
        }
        self.details.setPlainText(
            f"{len(ids)} seleccionada(s)\n{r['path']}\n\nFormato: {r['codec']} · {r['bits'] or '—'} bits\nMotor: {r['feature_version'] or 'sin analizar'}\n\n"
            + json.dumps(compact, ensure_ascii=False, indent=2)
            + "\n\n"
            + (r["error"] or "")
            + "\n\nBPM puede estar a mitad/doble. RMS no equivale a LUFS. Los cortes no prueban recompresión. Cada botón de género guarda el cambio inmediatamente y conserva los otros géneros."
        )

    def update_genre_buttons(self, ids):
        self.update_suggestions(ids)
        label_sets = [set(json.loads(self.cache[ident]["labels"])) for ident in ids]
        for name, button in self.genre_buttons.items():
            assigned = sum(name in labels for labels in label_sets)
            button.setEnabled(bool(ids))
            button.setChecked(bool(ids) and assigned == len(ids))
            button.setToolTip(
                f"{assigned}/{len(ids)} seleccionadas. Pulsa para añadir; si todas lo tienen, se quita."
            )

    def toggle_genre(self, name, checked=False):
        ids = self.selected_ids()
        if not ids or (self.worker and self.worker.isRunning()):
            return
        self.lib.toggle_genre(ids, name)
        self.update_track_rows(ids)
        self.log.append(
            "Género guardado. Reentrena cuando quieras actualizar el clasificador."
        )

    def update_track_rows(self, ids):
        affected = set(ids)
        self.refreshing = True
        self.table.setSortingEnabled(False)
        for row in range(self.table.rowCount()):
            ident = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            if ident in affected:
                self.cache[ident] = self.lib.track(ident)
                self.populate_row(row, self.cache[ident])
                for column in range(self.table.columnCount()):
                    self.table.item(row, column).setSelected(True)
        self.table.setSortingEnabled(True)
        self.refreshing = False
        self.apply_filter()
        self.update_stats()
        self.select()

    def update_stats(self):
        reviewed = sum(row["reviewed"] for row in self.cache.values())
        self.stats.setText(
            f"{len(self.cache)} canciones · {reviewed} revisadas\nDatos: {self.root}"
        )

    def add_style(self):
        name, ok = QInputDialog.getText(self, "Nuevo estilo", "Nombre:")
        if ok:
            try:
                self.lib.add_style(name)
                self.reload_styles()
                self.select()
            except Exception as e:
                QMessageBox.warning(self, "Estilo", str(e))

    def rename_style(self):
        name = self.styles.currentText()
        if not name:
            return
        name, ok = QInputDialog.getText(
            self, "Renombrar estilo", "Nuevo nombre:", text=name
        )
        if ok:
            try:
                self.lib.rename_style(self.styles.currentText(), name)
                self.reload_styles()
                self.refresh()
                self.log.append(
                    "Estilo renombrado. Reentrena antes de obtener nuevas sugerencias."
                )
            except Exception as e:
                QMessageBox.warning(self, "Estilo", str(e))

    def import_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Carpeta de música")
        if folder:
            self.start("scan", folder)

    def start(self, action, argument=None):
        if self.worker and self.worker.isRunning():
            return
        if action == "predict":
            dirty = self.lib.db.execute(
                "SELECT value FROM settings WHERE key='model_dirty'"
            ).fetchone()
            if dirty and dirty[0] == "1":
                QMessageBox.information(
                    self,
                    "Actualizar modelo",
                    "Hay etiquetas nuevas. Pulsa Entrenar con mis etiquetas.",
                )
                return
        self.worker = Worker(
            self.root,
            action,
            argument,
            self.engine_choices.checkedAction().data(),
        )
        for b in self.busy_controls:
            b.setEnabled(False)
        self.engine.setEnabled(False)
        self.pause.setEnabled(action in ("scan", "analyze"))
        self.worker.message.connect(self.log.append)
        self.worker.done.connect(self.log.append)
        self.worker.failed.connect(self.error)
        self.worker.finished.connect(self.finished)
        self.worker.start()
        self.update_suggestions(self.selected_ids())

    def error(self, text):
        self.log.append(text)
        QMessageBox.warning(self, "No se pudo completar", text.split("\nTraceback")[0])

    def finished(self):
        for b in self.busy_controls:
            b.setEnabled(True)
        self.engine.setEnabled(True)
        self.pause.setEnabled(False)
        self.refresh()

    def stop_work(self):
        if self.worker:
            self.worker.stop.set()
            self.log.append("Pausa solicitada; terminando el archivo actual…")

    def play(self, *args):
        if self.current is None:
            return
        r = self.lib.track(self.current)
        if self.loaded != self.current:
            self.loaded = self.current
            self.chart_request = None
            self.player.setSource(QUrl.fromLocalFile(r["path"]))
            self.player.play()
            self.show_chart()
            self.update_now_playing()
        elif self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    @staticmethod
    def track_name(row):
        title = row["title"] or Path(row["path"]).stem
        artist = row["artist"] or ""
        if artist:
            return f"{artist} - {title}"
        return title

    def update_now_playing(self, *args):
        row = self.lib.track(self.loaded) if self.loaded is not None else None
        if row is None:
            text = "Ahora sonando: ninguna canción"
        else:
            state = self.player.playbackState()
            prefix = "Ahora sonando"
            if state == QMediaPlayer.PlaybackState.PausedState:
                prefix = "En pausa"
            elif state == QMediaPlayer.PlaybackState.StoppedState:
                prefix = "Detenida"
            text = f"{prefix}: {self.track_name(row)}"
        self.now_playing.setText(text)
        self.now_playing.setToolTip(text)

    def search_genre_online(self):
        ids = self.selected_ids()
        if len(ids) != 1:
            return
        row = self.lib.track(ids[0])
        query = f'de qué género es la canción "{self.track_name(row)}"'
        url = QUrl("https://www.google.com/search?" + urlencode({"q": query}))
        if not QDesktopServices.openUrl(url):
            self.statusBar().showMessage(
                "No se pudo abrir el navegador predeterminado."
            )

    def position(self, ms):
        self.timeline.set_position(ms)
        elapsed = self.format_time(ms)
        duration = self.format_time(self.player.duration())
        self.time.setText(f"{elapsed} / {duration}")

    @staticmethod
    def format_time(milliseconds):
        seconds = int(milliseconds) // 1000
        return f"{seconds // 60}:{seconds % 60:02d}"

    def seek_changed(self, value):
        if self.loaded is not None:
            self.player.setPosition(value)
            self.timeline.set_position(value)

    def show_chart(self):
        if self.loaded is None:
            return
        row = self.lib.track(self.loaded)
        if row is None:
            return
        request = dict(path=row["path"], id=row["id"], duration=row["duration"] or 0)
        if request == self.chart_request:
            return
        self.chart_request = request
        self.chart_pending = request
        if self.chart_worker is not None:
            self.chart_worker.stop.set()
        self.timeline.clear("Cargando forma de onda y espectrograma completos…")
        self.timeline.set_duration(round(request["duration"] * 1000))
        self.chart_timer.start()

    def load_chart(self):
        if self.chart_worker is not None or self.chart_pending is None:
            return
        request = self.chart_pending
        self.chart_pending = None
        self.chart_worker = ChartWorker(request)
        self.chart_worker.ready.connect(self.receive_chart)
        self.chart_worker.failed.connect(self.chart_failed)
        self.chart_worker.finished.connect(self.chart_finished)
        self.chart_worker.start()

    def receive_chart(self, data):
        identity = {key: data[key] for key in ("path", "id", "duration")}
        if identity == self.chart_request and data["id"] == self.loaded:
            self.timeline.set_data(data["overview"])
            self.timeline.set_position(self.player.position())

    def chart_failed(self, message):
        if self.chart_worker.request == self.chart_request:
            self.log.append("No se pudo cargar la vista de audio: " + message)
            self.timeline.clear("No se pudo cargar el audio. Consulta el registro.")

    def chart_finished(self):
        self.chart_worker.deleteLater()
        self.chart_worker = None
        self.load_chart()

    def choose_columns(self):
        menu = QMenu(self)
        for column in range(self.table.columnCount()):
            action = menu.addAction(self.table.horizontalHeaderItem(column).text())
            action.setCheckable(True)
            action.setChecked(not self.table.isColumnHidden(column))
            action.setData(column)
        menu.triggered.connect(self.toggle_column)
        menu.exec(self.cursor().pos())

    def toggle_column(self, action):
        self.table.setColumnHidden(action.data(), not action.isChecked())
        hidden = [
            column
            for column in range(self.table.columnCount())
            if self.table.isColumnHidden(column)
        ]
        with self.lib.db:
            self.lib.db.execute(
                "INSERT OR REPLACE INTO settings VALUES ('hidden_columns', ?)",
                (json.dumps(hidden),),
            )

    def restore_columns(self):
        setting = self.lib.db.execute(
            "SELECT value FROM settings WHERE key='hidden_columns'"
        ).fetchone()
        if setting:
            for column in json.loads(setting[0]):
                if isinstance(column, int) and 0 <= column < self.table.columnCount():
                    self.table.setColumnHidden(column, True)

    def delete_style(self):
        name = self.styles.currentText()
        if not name:
            return
        answer = QMessageBox.question(
            self,
            "Eliminar género",
            f"¿Eliminar «{name}»? Sus canciones quedarán sin revisar y conservarán sus otros géneros.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.lib.delete_style(name)
        self.reload_styles()
        self.refresh()
        self.log.append(
            "Género eliminado. Reentrena el clasificador para actualizar las sugerencias."
        )

    def delete_tracks(self):
        ids = self.selected_ids()
        if not ids:
            return
        paths = [self.lib.track(ident)["path"] for ident in ids]
        dialog = QMessageBox(
            QMessageBox.Icon.Warning,
            "Eliminar archivos definitivamente",
            f"¿Eliminar {len(ids)} archivo(s) del dispositivo? No se enviarán a la papelera.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            self,
        )
        dialog.setDetailedText("\n".join(paths))
        dialog.setDefaultButton(QMessageBox.StandardButton.No)
        if dialog.exec() != QMessageBox.StandardButton.Yes:
            return
        if self.chart_worker is not None and self.chart_worker.request["id"] in ids:
            QMessageBox.information(
                self,
                "Vista de audio en curso",
                "Espera a que termine de cargar la vista y vuelve a eliminar.",
            )
            return
        if self.loaded in ids:
            self.chart_timer.stop()
            self.chart_pending = None
            self.chart_request = None
            self.player.stop()
            self.player.setSource(QUrl())
            self.loaded = None
            self.update_now_playing()
            self.timeline.clear()
        failures = []
        deleted = []
        for ident in ids:
            try:
                self.lib.delete_track_file(ident)
                deleted.append(ident)
            except (OSError, ValueError) as error:
                failures.append(f"{self.lib.track(ident)['path']}: {error}")
        self.remove_track_rows(deleted)
        if failures:
            self.error("\n".join(failures))

    def remove_track_rows(self, ids):
        if not ids:
            return
        deleted = set(ids)
        self.refreshing = True
        self.table.setUpdatesEnabled(False)
        self.table.setSortingEnabled(False)
        try:
            for row in range(self.table.rowCount() - 1, -1, -1):
                ident = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
                if ident in deleted:
                    self.table.removeRow(row)
                    self.cache.pop(ident, None)
                else:
                    # The classifier is stale after deletion; update only this cell.
                    self.cache[ident]["predictions"] = "[]"
                    item = self.table.item(row, 2)
                    item.setText("")
                    item.sort_value = -1
        finally:
            self.table.setSortingEnabled(True)
            self.table.setUpdatesEnabled(True)
            self.refreshing = False
        self.update_stats()
        self.select()

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar biblioteca", "biblioteca.csv", "CSV (*.csv)"
        )
        if path:
            try:
                self.lib.export(path)
                self.log.append("CSV exportado.")
            except Exception as e:
                self.error(str(e))

    def backup(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Copia de etiquetas y análisis",
            "biblioteca-backup.sqlite3",
            "SQLite (*.sqlite3)",
        )
        if path:
            try:
                if Path(path).resolve() == (self.root / "library.sqlite3").resolve():
                    raise ValueError("Elige otra ruta para la copia.")
                self.lib.backup(path)
                self.log.append(
                    "Copia creada. El modelo puede regenerarse con Entrenar."
                )
            except Exception as e:
                self.error(str(e))

    def model_report(self):
        path = self.root / "personal-model.json"
        if not path.exists():
            QMessageBox.information(
                self, "Modelo", "Todavía no hay un modelo entrenado."
            )
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Informe del clasificador")
        dialog.resize(660, 550)
        layout = QVBoxLayout(dialog)
        view = QTextEdit()
        view.setReadOnly(True)
        view.setPlainText(
            json.dumps(
                json.loads(path.read_text(encoding="utf-8"))["report"],
                ensure_ascii=False,
                indent=2,
            )
        )
        layout.addWidget(view)
        dialog.exec()

    def closeEvent(self, event):
        if self.chart_worker is not None:
            QMessageBox.information(
                self,
                "Vista de audio en curso",
                "Espera a que termine de cargar la vista antes de cerrar.",
            )
            event.ignore()
            return
        self.chart_timer.stop()
        if self.worker and self.worker.isRunning():
            self.stop_work()
            QMessageBox.information(
                self,
                "Trabajo en curso",
                "Se ha solicitado pausa. Espera a que termine la tarea actual antes de cerrar.",
            )
            event.ignore()
            return
        self.player.stop()
        self.player.setSource(QUrl())
        self.lib.close()
        event.accept()


if __name__ == "__main__":
    application = QApplication(sys.argv)
    application.setApplicationName("ElectronicaLab")
    window = Window()
    window.show()
    sys.exit(application.exec())
