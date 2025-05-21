import sys
import time
import os
import platform
import pathlib
import ctypes
import numpy as np
import imageio.v2 as imageio   # NEW – for optional PNG export
import matplotlib.pyplot as plt      # NEW – annotate PNG exports
import librosa

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QProgressDialog,
    QDialog,
    QVBoxLayout,
    QPushButton,
    QHBoxLayout,
    QCheckBox,               # NEW
    QLabel,                  # NEW
    QSlider,                 # NEW
)
from PySide6.QtCore import Qt
import pyqtgraph as pg
from pyqtgraph import PlotWidget, ImageItem

# suppress Windows loader error dialogs (Bad Image, missing DLL pop-ups) on Windows only
if platform.system() == "Windows":
    SEM_FAILCRITICALERRORS   = 0x0001
    SEM_NOGPFAULTERRORBOX    = 0x0002
    SEM_NOOPENFILEERRORBOX   = 0x8000
    ctypes.windll.kernel32.SetErrorMode(
        SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX
    )

# ── Windows‑only auto‑update (ctypes + WinSparkle DLL) ───────────────────
if platform.system() == "Windows":
    try:
        dll_path = pathlib.Path(sys.executable).with_name("winsparkle.dll")
        _ws = ctypes.WinDLL(str(dll_path))
        _ws.win_sparkle_set_appcast_url.argtypes = [ctypes.c_wchar_p]
        _ws.win_sparkle_set_appcast_url("https://github.com/Lixin-TU/AudioSpectroDemo/blob/main/appcast.xml")
        _ws.win_sparkle_set_app_details.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_wchar_p]
        _ws.win_sparkle_set_app_details("UBCO-ISDPRL", "AudioSpectroDemo", "0.2.6")
        # Verbose logging (0=errors, 1=info, 2=debug)
        _ws.win_sparkle_set_verbosity_level.argtypes = [ctypes.c_int]
        _ws.win_sparkle_set_verbosity_level(2)
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

# ── Mel‑scale helpers ──────────────────────────────────────────────────
MEL_MAX = librosa.hz_to_mel(MAX_FREQ)  # upper mel corresponding to MAX_FREQ

# Plasma‑style dark‑purple → yellow ramp (closely matches reference)
COLMAP_COLORS = [
    (13, 8, 135),
    (63, 0, 153),
    (106, 0, 168),
    (162, 0, 167),
    (203, 71, 120),
    (229, 107, 53),
    (248, 148, 65),
    (254, 197, 34),
    (253, 245, 2),
]

# Smooth 256‑step plasma lookup table for better visualisation
LUT = (plt.get_cmap("plasma")(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

def _hz_to_k_label(hz: int) -> str:
    return "0" if hz == 0 else f"{int(round(hz / 1000))}"

# ── PNG/export dimensions ───────────────────────────────────────────────
EXPORT_W = 800    # pixels width
EXPORT_H = 400    # pixels height


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AudioSpectroDemo")
        self.resize(900, 600)

        # ── Main UI: button + optional export checkbox ──
        central = pg.QtWidgets.QWidget(self)
        layout = QVBoxLayout(central)

        # App info banner
        self.info_label = QLabel("UBCO‑ISDPRL  •  AudioSpectroDemo v0.2.6")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)
        self.info_label.setText(self.info_label.text() + "  •  Updater=DEBUG")

        self.open_btn = QPushButton("Open WAV files")
        self.export_checkbox = QCheckBox("Export spectrograms")  # default: unchecked
        self.export_checkbox.setChecked(False)

        layout.addWidget(self.open_btn)
        layout.addWidget(self.export_checkbox)
        layout.addStretch(1)

        self.setCentralWidget(central)

        # Connect
        self.open_btn.clicked.connect(self.open_wav)

    def open_wav(self):
        dlg = QFileDialog(self)
        dlg.setFileMode(QFileDialog.ExistingFiles)
        dlg.setNameFilter("WAV (*.wav *.WAV)")
        if not dlg.exec():
            return
        files = dlg.selectedFiles()
        if not files:
            return

        # Precompute spectrograms with progress
        progress = QProgressDialog("Processing audio...", None, 0, len(files), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.show()
        QApplication.processEvents()
        spectrograms = []
        for i, file_path in enumerate(files):
            progress.setValue(i)
            QApplication.processEvents()
            img = wav_to_mel_image(file_path)
            spectrograms.append((file_path, img))
        # Optionally export each spectrogram as a PNG alongside its source WAV
        if self.export_checkbox.isChecked():
            for wav_path, img in spectrograms:
                img_norm = img - img.min()
                if img_norm.max() > 0:
                    img_norm = img_norm / img_norm.max()

                # Map to colour LUT (same as viewer)
                indices = (img_norm * 255).astype(np.uint8)
                lut = LUT

                rgb_img = lut[indices]  # (H, W, 3)

                # Build axes extents
                hop = 512
                # Save PNGs in a sibling folder called “spectrograms”
                export_dir = os.path.join(os.path.dirname(wav_path), "spectrograms")
                os.makedirs(export_dir, exist_ok=True)

                duration_min = rgb_img.shape[1] * hop / TARGET_SR / 60.0

                # Plot with axes & title
                # Fixed export canvas
                fig = plt.figure(figsize=(EXPORT_W / 300, EXPORT_H / 300), dpi=300)
                ax = fig.add_subplot(111)
                # Reduce default margins so the image fills more of the canvas
                fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12)
                ax.imshow(
                    np.flipud(rgb_img),
                    aspect='auto',
                    extent=[0, duration_min, 0, MAX_FREQ],
                    origin='lower'
                )
                ax.set_xlabel("Time (min)", fontsize=3)
                ax.set_ylabel("Frequency (kHz)", fontsize=3)
                major_freqs = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000]
                ax.set_yticks(major_freqs)
                ax.set_yticklabels([_hz_to_k_label(f) for f in major_freqs], fontsize=3)
                ax.set_ylim(0, MAX_FREQ)
                # Thin, inward‑facing light‑grey ticks
                ax.tick_params(
                    axis="both",
                    which="both",
                    direction="in",
                    color="0.6",     # light gray
                    width=0.4,
                    length=3,
                    labelsize=3,
                )
                for spine in ax.spines.values():
                    spine.set_linewidth(0.4)
                    spine.set_color("0.6")
                # ax.tick_params(axis="x", labelsize=3)  # replaced by above
                ax.title.set_fontsize(5)
                ax.set_title(os.path.basename(wav_path), fontsize=5, pad=4)
                png_path = os.path.join(
                    export_dir,
                    os.path.splitext(os.path.basename(wav_path))[0] + ".png"
                )
                fig.savefig(
                    png_path,
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0.01,
                )
                plt.close(fig)
        progress.close()

        # Create viewer dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Spectrogram Viewer")
        main_layout = QVBoxLayout(dialog)

        # Plot area
        pw = PlotWidget(background="w")
        main_layout.addWidget(pw)

        # Navigation buttons
        btn_layout = QHBoxLayout()
        prev_btn = QPushButton("Previous")
        next_btn = QPushButton("Next")
        btn_layout.addWidget(prev_btn)
        btn_layout.addWidget(next_btn)
        main_layout.addLayout(btn_layout)

        # Gain (0–40 dB) slider + readout (default 0 dB)
        gain_layout = QHBoxLayout()
        gain_label_prefix = QLabel("Color gain (dB):")
        gain_slider = QSlider(Qt.Horizontal)
        gain_slider.setRange(0, 40)        # 0 – 40 dB range as requested
        gain_slider.setValue(0)            # default 0 dB
        gain_value = QLabel("0 dB")
        gain_layout.addWidget(gain_label_prefix)
        gain_layout.addWidget(gain_slider, 1)
        gain_layout.addWidget(gain_value)
        main_layout.addLayout(gain_layout)

        small_font = self.font()
        small_font.setPointSizeF(small_font.pointSizeF() * 0.85)
        gain_label_prefix.setFont(small_font)
        gain_value.setFont(small_font)

        from PySide6.QtGui import QTransform

        index = 0

        def update_view(idx):
            pw.clear()
            file_path, img = spectrograms[idx]
            gain_db = gain_slider.value()
            # map uint8→float, apply gain as 10^(dB/20), then re‑quantise
            img_float = img.astype(np.float32) / 255.0
            factor = 10 ** (gain_db / 20.0)
            img_float = np.clip(img_float * factor, 0.0, 1.0)
            img_disp = (img_float * 255.0).astype(np.uint8)
            dialog.setWindowTitle(os.path.basename(file_path))

            freqs = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000]
            y_ticks = [(f, _hz_to_k_label(f)) for f in freqs]
            n_bins = img.shape[0]  # keep for scale calc below

            hop = 512
            scale_x = hop / TARGET_SR / 60

            pw.setLabel("bottom", "Time", units="min")
            pw.setLabel("left", "Frequency", units="Hz")
            pw.getAxis("left").setTicks([y_ticks])

            img_item = ImageItem(img_disp)
            img_item.setLookupTable(LUT, update=True)
            img_item.setOpts(axisOrder="row-major")          # no invert
            # Scale X (time) and Y (frequency) axes; flip Y by using a negative scale
            img_item.setTransform(
                QTransform().scale(scale_x, -MAX_FREQ / (n_bins - 1))
            )
            # After negative scale, shift the image up so 0 Hz is at the bottom
            img_item.setPos(0, MAX_FREQ)
            pw.addItem(img_item)
            pw.setLimits(yMin=0, yMax=MAX_FREQ)

            prev_btn.setEnabled(idx > 0)
            next_btn.setEnabled(idx < len(spectrograms) - 1)

        def on_gain_change(v):
            gain_value.setText(f"{int(v)} dB")
            update_view(index)
        gain_slider.valueChanged.connect(on_gain_change)

        # Button callbacks
        prev_btn.clicked.connect(lambda: update_view_wrapper(index - 1))
        next_btn.clicked.connect(lambda: update_view_wrapper(index + 1))

        def update_view_wrapper(new_idx):
            nonlocal index
            index = new_idx
            update_view(index)

        # Initialize view
        update_view(index)

        dialog.resize(EXPORT_W, EXPORT_H + 120)  # keep extra space for buttons
        dialog.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Main()
    window.show()
    if platform.system() == "Windows" and _ws:
        _ws.win_sparkle_cleanup()
    sys.exit(app.exec())