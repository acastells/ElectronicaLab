import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import json
import tempfile
import unittest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app import Window, Worker


class UiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_labels_filter_and_worker_completion(self):
        with tempfile.TemporaryDirectory() as root:
            w = Window(root)
            with w.lib.db:
                w.lib.db.execute(
                    "INSERT INTO tracks(path,title,artist) VALUES ('x.wav','Demo','Test')"
                )
            w.refresh()
            w.table.selectRow(0)
            w.select()
            w.genre_buttons["Dubstep"].click()
            self.assertEqual(json.loads(w.lib.track(1)["labels"]), ["Dubstep"])
            w.search.setText("absent")
            self.assertTrue(w.table.isRowHidden(0))
            w.search.clear()
            self.assertFalse(w.table.isRowHidden(0))
            worker = Worker(root, "scan", root)
            messages = []
            worker.done.connect(messages.append)
            worker.start()
            self.assertTrue(worker.wait(10000))
            self.app.processEvents()
            self.assertTrue(messages)
            from pathlib import Path

            import numpy as np
            from scipy.io.wavfile import write

            tone = Path(root) / "tone.wav"
            sr = 44100
            write(
                tone,
                sr,
                (np.sin(np.arange(sr * 3) / sr * 2 * np.pi * 440) * 16000).astype(
                    np.int16
                ),
            )
            w.lib.scan(root)
            worker2 = Worker(root, "analyze")
            worker2.start()
            self.assertTrue(worker2.wait(45000))
            self.app.processEvents()
            analyzed = [r for r in w.lib.rows() if r["path"] == str(tone)]
            self.assertEqual(analyzed[0]["status"], "done", analyzed[0]["error"])
            w.close()

    def test_click_slider_and_drag(self):
        from PySide6.QtCore import QPoint
        from PySide6.QtTest import QTest

        from app import ClickSlider

        slider = ClickSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.resize(200, 30)
        slider.show()
        QTest.mouseClick(slider, Qt.MouseButton.LeftButton, pos=QPoint(100, 15))
        self.assertAlmostEqual(slider.value(), 50, delta=2)
        QTest.mousePress(slider, Qt.MouseButton.LeftButton, pos=QPoint(100, 15))
        QTest.mouseMove(slider, QPoint(160, 15))
        QTest.mouseRelease(slider, Qt.MouseButton.LeftButton, pos=QPoint(160, 15))
        self.assertGreater(slider.value(), 75)
        self.assertFalse(slider.isSliderDown())
        slider.close()

    def test_sort_identity_and_column_persistence(self):
        from PySide6.QtGui import QAction

        with tempfile.TemporaryDirectory() as root:
            window = Window(root)
            with window.lib.db:
                window.lib.db.executemany(
                    "INSERT INTO tracks(path,title,artist,bitrate) VALUES (?,?,?,?)",
                    [("a.wav", "A", "", 100), ("b.wav", "B", "", 20)],
                )
            window.refresh()
            window.table.sortItems(4, Qt.SortOrder.AscendingOrder)
            window.table.selectRow(0)
            self.assertEqual(window.selected_ids(), [2])
            self.assertEqual(window.table.item(0, 8).text(), "b.wav")
            window.refresh()
            self.assertEqual(window.selected_ids(), [2])
            action = QAction("path")
            action.setCheckable(True)
            action.setData(8)
            window.toggle_column(action)
            window.close()
            reopened = Window(root)
            self.assertTrue(reopened.table.isColumnHidden(8))
            reopened.close()

    def test_play_loads_chart_without_analysis_worker(self):
        import time
        from pathlib import Path

        import numpy as np
        from scipy.io.wavfile import write

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "tone.wav"
            write(path, 48000, np.sin(np.arange(48000) * 0.1).astype(np.float32))
            window = Window(root)
            window.lib.scan(root)
            window.refresh()
            window.table.selectRow(0)
            window.play()
            deadline = time.monotonic() + 10
            while window.timeline.data is None and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.01)
            self.assertIsNotNone(window.timeline.data)
            self.assertAlmostEqual(window.timeline.data["duration"], 1, places=2)
            self.assertIsNone(window.worker)
            while window.chart_worker is not None and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.01)
            window.close()

    def test_waveform_seek_and_cursor_reuses_background(self):
        import numpy as np
        from PySide6.QtCore import QPoint
        from PySide6.QtTest import QTest

        from timeline import AudioTimeline

        timeline = AudioTimeline()
        timeline.resize(800, 270)
        timeline.set_data(
            {
                "peaks": np.ones(100),
                "spectrum_db": np.zeros((100, 100)),
                "duration": 120,
            }
        )
        timeline.show()
        self.app.processEvents()
        background = timeline.background
        positions = []
        timeline.seek_requested.connect(positions.append)
        midpoint = timeline.LEFT + timeline.plot_width() // 2
        QTest.mouseClick(timeline, Qt.MouseButton.LeftButton, pos=QPoint(midpoint, 65))
        self.assertAlmostEqual(positions[-1], 60000, delta=100)
        timeline.set_position(60000)
        self.app.processEvents()
        self.assertIs(timeline.background, background)
        timeline.close()

    def test_delete_keeps_other_items_and_does_not_refresh(self):
        from pathlib import Path
        from unittest.mock import patch

        from PySide6.QtWidgets import QHeaderView, QMessageBox

        with tempfile.TemporaryDirectory() as root:
            window = Window(root)
            for title in ["A", "B"]:
                path = Path(root) / (title + ".wav")
                path.write_bytes(b"test")
                with window.lib.db:
                    window.lib.db.execute(
                        "INSERT INTO tracks(path,title,artist) VALUES (?,?,?)",
                        (str(path), title, ""),
                    )
            window.refresh()
            self.assertEqual(
                window.table.horizontalHeader().sectionResizeMode(0),
                QHeaderView.ResizeMode.Interactive,
            )
            window.table.setColumnWidth(0, 420)
            survivor = window.table.item(1, 0)
            window.table.selectRow(0)
            with (
                patch.object(
                    QMessageBox, "exec", return_value=QMessageBox.StandardButton.Yes
                ),
                patch.object(window, "refresh") as refresh,
            ):
                window.delete_tracks()
                refresh.assert_not_called()
            self.assertEqual(window.table.rowCount(), 1)
            self.assertIs(window.table.item(0, 0), survivor)
            self.assertEqual(window.table.columnWidth(0), 420)
            self.assertFalse((Path(root) / "A.wav").exists())
            window.close()

    def test_genre_click_preserves_other_labels(self):
        with tempfile.TemporaryDirectory() as root:
            window = Window(root)
            with window.lib.db:
                window.lib.db.execute(
                    "INSERT INTO tracks(path,title,artist,labels,reviewed) VALUES ('a','A','','[\"House\"]',1)"
                )
            window.refresh()
            window.table.selectRow(0)
            window.genre_buttons["Dubstep"].click()
            self.assertEqual(
                json.loads(window.lib.track(1)["labels"]), ["Dubstep", "House"]
            )
            window.genre_buttons["Dubstep"].click()
            self.assertEqual(json.loads(window.lib.track(1)["labels"]), ["House"])
            window.genre_buttons["House"].click()
            self.assertEqual(window.lib.track(1)["reviewed"], 0)
            window.close()

    def test_menus_and_google_use_selected_song(self):
        from unittest.mock import patch
        from urllib.parse import parse_qs, urlparse

        from PySide6.QtWidgets import QPushButton

        with tempfile.TemporaryDirectory() as root:
            window = Window(root)
            self.assertEqual(
                [action.text() for action in window.menuBar().actions()],
                ["Biblioteca", "Análisis", "Clasificador", "Ver"],
            )
            button_names = [
                button.text() for button in window.findChildren(QPushButton)
            ]
            self.assertNotIn("Exportar CSV", button_names)
            self.assertNotIn("Analizar / continuar", button_names)
            self.assertFalse(window.google_button.isEnabled())
            with window.lib.db:
                window.lib.db.executemany(
                    "INSERT INTO tracks(path,title,artist) VALUES (?,?,?)",
                    [("a.wav", "A & B #1", "Artista ñ"), ("b.wav", "Otra", "")],
                )
            window.refresh()
            window.loaded = 2
            window.table.selectRow(0)
            with patch("app.QDesktopServices.openUrl", return_value=True) as open_url:
                window.google_button.click()
                url = open_url.call_args.args[0].toString()
                query = parse_qs(urlparse(url).query)["q"][0]
                self.assertEqual(
                    query, 'de qué género es la canción "Artista ñ - A & B #1"'
                )
            window.update_now_playing()
            self.assertIn("Otra", window.now_playing.text())
            self.assertNotIn("Artista", window.now_playing.text())
            window.table.clearSelection()
            self.assertFalse(window.google_button.isEnabled())
            window.close()

    def test_panels_persist_and_top_suggestion_assigns(self):
        with tempfile.TemporaryDirectory() as root:
            window = Window(root)
            self.assertTrue(window.log.isHidden())
            self.assertTrue(window.details.isHidden())
            window.panel_actions["activity"].setChecked(True)
            self.assertFalse(window.log.isHidden())
            with window.lib.db:
                window.lib.db.execute(
                    "INSERT INTO tracks(path,title,artist,labels,predictions) VALUES (?,?,?,?,?)",
                    (
                        "a.wav",
                        "A",
                        "",
                        '["House"]',
                        json.dumps(
                            [
                                ["Techno", 0.6],
                                ["Dubstep", 0.95],
                                ["Trance", 0.8],
                                ["EDM", 0.2],
                            ]
                        ),
                    ),
                )
            window.refresh()
            window.table.selectRow(0)
            self.assertEqual(
                [name for name, _ in window.visible_suggestions],
                ["Dubstep", "Trance", "Techno"],
            )
            window.suggestion_buttons[0].click()
            self.assertEqual(
                json.loads(window.lib.track(1)["labels"]), ["Dubstep", "House"]
            )
            self.assertEqual(window.lib.track(1)["reviewed"], 1)
            window.close()
            reopened = Window(root)
            self.assertFalse(reopened.log.isHidden())
            self.assertTrue(reopened.details.isHidden())
            reopened.close()
