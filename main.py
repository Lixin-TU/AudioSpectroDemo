import ctypes
# suppress Windows loader error dialogs (Bad Image, missing DLL pop-ups) on Windows only
import platform
if platform.system() == "Windows":
    SEM_FAILCRITICALERRORS   = 0x0001
    SEM_NOGPFAULTERRORBOX    = 0x0002
    SEM_NOOPENFILEERRORBOX   = 0x8000
    ctypes.windll.kernel32.SetErrorMode(
        SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX
    )
import sys
import time
import os
import numpy as np
import librosa
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QProgressDialog,
)
from PySide6.QtCore import Qt
import pyqtgraph as pg
from pyqtgraph import PlotWidget, ImageItem
from pyqtgraph.exporters import ImageExporter

# ── Windows‑only auto‑update (ctypes + WinSparkle DLL) ───────────────────
import ctypes, platform, pathlib, sys
if platform.system() == "Windows":
    try:
        dll_path = pathlib.Path(sys.executable).with_name("winsparkle.dll")
        _ws = ctypes.WinDLL(str(dll_path))
        _ws.win_sparkle_set_appcast_url.argtypes = [ctypes.c_wchar_p]
        _ws.win_sparkle_set_appcast_url("https://Lixin-TU.github.io/AudioSpectroDemo/appcast.xml")
        _ws.win_sparkle_init()
        _ws.win_sparkle_check_update_without_ui()
    except OSError:
        # DLL missing or wrong arch – skip updater but keep app running
        _ws = None
# ───────────────────────────────────────────────────────────────────────────

from dsp.mel import (
    wav_to_mel_image,
    TARGET_SR,
    MIN_FREQ,
    MAX_FREQ,
)

EXPORT_W = 1600   # pixels
EXPORT_H = 600    # pixels


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AudioSpectroDemo")
        self.resize(900, 600)

        # File → Export PNG
        file_menu = self.menuBar().addMenu("File")
        export_act = file_menu.addAction("Export PNG…")
        export_act.triggered.connect(self.export_png)

        self.open_wav()

    def open_wav(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open WAV file",
            filter="WAV (*.wav *.WAV)",
        )
        if not file_path:
            return

        self.setWindowTitle(os.path.basename(file_path))

        progress = QProgressDialog("Processing audio...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()
        try:
            time.sleep(2)
            img = wav_to_mel_image(file_path)  # numpy uint8 0-255
        finally:
            progress.close()

        mel_min = librosa.hz_to_mel(0)             # include DC
        mel_max = librosa.hz_to_mel(MAX_FREQ)

        # Frequency ticks every 2 kHz (0 – 16 kHz)
        n_bins = img.shape[0]

        def hz_to_row(hz):
            mel = librosa.hz_to_mel(hz)
            rel = (mel - mel_min) / (mel_max - mel_min)
            return int(round(rel * (n_bins - 1)))

        desired_freqs = list(range(0, 18001, 2000))  # 0,2k,…16k Hz
        y_ticks = [(hz_to_row(f),
                    f"{f//1000}k" if f >= 1000 else str(f))
                   for f in desired_freqs]
        y_ticks.sort(key=lambda t: t[0])

        hop = 1024                           # must match mel.py default
        scale_x = hop / TARGET_SR / 60       # minutes per pixel (x)

        pw = PlotWidget(background="w")
        pw.setLabel("bottom", "Time", units="min")
        pw.setLabel("left", "Frequency", units="Hz")
        pw.getAxis("left").setTicks([y_ticks])

        from PySide6.QtGui import QTransform

        img_item = ImageItem(img)

        # Dark roseus colormap (black → blue → magenta)
        colors = [
            (0,   0,   0),
            (0,   0,  80),
            (64,  0, 128),
            (192, 0, 192),
            (255, 128, 255),
        ]
        positions = np.linspace(0.0, 1.0, len(colors))
        cmap = pg.ColorMap(positions, colors)
        lut = cmap.getLookupTable(0.0, 1.0, 256).astype(np.ubyte)
        img_item.setLookupTable(lut, update=True)

        img_item.setOpts(axisOrder="row-major")
        img_item.setTransform(QTransform().scale(scale_x, 1.0))
        img_item.setPos(0, 0)                # origin at (0,0)
        pw.addItem(img_item)
        pw.setLimits(yMin=0, yMax=MAX_FREQ)

        self.view = pw
        self.setCentralWidget(pw)

    # ── Export PNG at fixed size ────────────────────────────────────────
    def export_png(self):
        """
        Export the currently displayed spectrogram as a PNG
        with fixed pixel dimensions defined by EXPORT_W / EXPORT_H.
        """
        if not hasattr(self, "view"):
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Spectrogram",
            filter="PNG (*.png)",
        )
        if not save_path:
            return

        exporter = ImageExporter(self.view.plotItem)
        exporter.params["width"] = EXPORT_W
        exporter.params["height"] = EXPORT_H
        exporter.export(save_path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Main()
    window.show()
    if platform.system() == "Windows" and _ws:
        _ws.win_sparkle_cleanup()
    sys.exit(app.exec())