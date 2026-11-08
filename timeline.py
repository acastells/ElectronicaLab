"""Cached whole-track waveform and spectrogram with a lightweight playhead."""

import numpy as np
from matplotlib import colormaps
from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget


class AudioTimeline(QWidget):
    seek_requested = Signal(int)
    LEFT = 48
    RIGHT = 12
    WAVE_TOP = 26
    WAVE_BOTTOM = 106
    SPEC_TOP = 132

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(270)
        self.setMaximumHeight(300)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.duration = 0
        self.position = 0
        self.data = None
        self.background = None
        self.spectrum_image = None
        self.message = "Selecciona una canción y pulsa reproducir."
        self.dragging = False

    def clear(self, message="Selecciona una canción y pulsa reproducir."):
        self.data = None
        self.spectrum_image = None
        self.message = message
        self.position = 0
        self.background = None
        self.update()

    def set_duration(self, milliseconds):
        self.duration = milliseconds

    def set_data(self, data):
        self.data = data
        self.duration = round(data["duration"] * 1000)
        colors = colormaps["magma"](np.clip((data["spectrum_db"] + 100) / 80, 0, 1))
        rgb = np.ascontiguousarray((colors[::-1, :, :3] * 255).astype(np.uint8))
        height, width, _ = rgb.shape
        self.spectrum_image = QImage(
            rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888
        ).copy()
        self.background = None
        self.update()

    def cursor_x(self):
        fraction = self.position / max(1, self.duration)
        return self.LEFT + round(np.clip(fraction, 0, 1) * self.plot_width())

    def plot_width(self):
        return max(1, self.width() - self.LEFT - self.RIGHT)

    def set_position(self, milliseconds):
        old_x = self.cursor_x()
        self.position = milliseconds
        new_x = self.cursor_x()
        self.update(QRect(old_x - 3, self.WAVE_TOP, 7, self.height()))
        self.update(QRect(new_x - 3, self.WAVE_TOP, 7, self.height()))

    def resizeEvent(self, event):
        self.background = None
        super().resizeEvent(event)

    def paintEvent(self, event):
        if self.background is None:
            self.build_background()
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.background)
        if self.data is not None:
            painter.setPen(QPen(QColor("#ffcf70"), 2))
            x = self.cursor_x()
            painter.drawLine(x, self.WAVE_TOP, x, self.height() - 24)

    def build_background(self):
        self.background = QPixmap(self.size())
        self.background.fill(QColor("#111d2b"))
        painter = QPainter(self.background)
        painter.setPen(QColor("#d0dce8"))
        if self.data is None:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)
            painter.end()
            return
        painter.drawText(
            self.LEFT,
            18,
            "FORMA DE ONDA · canción completa · clic o arrastra para avanzar",
        )
        painter.drawText(self.LEFT, 124, "ESPECTROGRAMA · canción completa")
        self.paint_waveform(painter)
        spectrum_rect = QRect(
            self.LEFT,
            self.SPEC_TOP,
            self.plot_width(),
            self.height() - self.SPEC_TOP - 24,
        )
        painter.drawImage(spectrum_rect, self.spectrum_image)
        painter.setPen(QColor("#d0dce8"))
        painter.drawText(2, self.SPEC_TOP + 10, "24 kHz")
        painter.drawText(12, spectrum_rect.bottom(), "0 Hz")
        for index in range(5):
            seconds = self.duration / 1000 * index / 4
            label = f"{int(seconds) // 60}:{int(seconds) % 60:02d}"
            x = self.LEFT + round(self.plot_width() * index / 4)
            painter.drawText(min(x, self.width() - 45), self.height() - 5, label)
        painter.end()

    def paint_waveform(self, painter):
        peaks = self.data["peaks"]
        scale = max(float(peaks.max()), 0.01)
        center = (self.WAVE_TOP + self.WAVE_BOTTOM) // 2
        height = (self.WAVE_BOTTOM - self.WAVE_TOP) / 2
        painter.setPen(QPen(QColor("#78e4c8"), 1))
        for pixel in range(self.plot_width()):
            start = min(len(peaks) - 1, pixel * len(peaks) // self.plot_width())
            end = max(start + 1, (pixel + 1) * len(peaks) // self.plot_width())
            amplitude = max(1, round(float(peaks[start:end].max()) / scale * height))
            x = self.LEFT + pixel
            painter.drawLine(x, center - amplitude, x, center + amplitude)

    def seek_pointer(self, event):
        if self.duration > 0:
            fraction = np.clip(
                (event.position().x() - self.LEFT) / self.plot_width(), 0, 1
            )
            self.seek_requested.emit(round(fraction * self.duration))
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.seek_pointer(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragging:
            self.seek_pointer(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.seek_pointer(event)

    def keyPressEvent(self, event):
        offsets = {Qt.Key.Key_Left: -5000, Qt.Key.Key_Right: 5000}
        if event.key() in offsets:
            target = max(0, min(self.duration, self.position + offsets[event.key()]))
            self.seek_requested.emit(target)
        else:
            super().keyPressEvent(event)
