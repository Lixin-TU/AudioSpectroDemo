import sys
import time
import os
import platform
import pathlib
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import librosa
import threading
import xml.etree.ElementTree as ET
import urllib.request
import subprocess
import tempfile
from datetime import datetime
import shutil

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QProgressDialog,
    QDialog,
    QVBoxLayout,
    QPushButton,
    QHBoxLayout,
    QCheckBox,
    QLabel,
    QSlider,
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject
import pyqtgraph as pg
from pyqtgraph import PlotWidget, ImageItem

# Global variables - will be set up in Main constructor
WINSPARKLE_LOG_PATH = None
_ws = None

# suppress Windows loader error dialogs
if platform.system() == "Windows":
    SEM_FAILCRITICALERRORS   = 0x0001
    SEM_NOGPFAULTERRORBOX    = 0x0002
    SEM_NOOPENFILEERRORBOX   = 0x8000
    ctypes.windll.kernel32.SetErrorMode(
        SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX
    )

# Update checker signal class
class UpdateChecker(QObject):
    update_available = Signal(dict)
    update_not_available = Signal(str)
    update_error = Signal(str)

# Global update checker instance
update_checker = UpdateChecker()

def parse_appcast_xml(url):
    """Parse the appcast XML to get update information"""
    try:
        with urllib.request.urlopen(url) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        items = root.findall('.//item')
        if not items:
            return None
        latest_item = items[0]

        version_el = latest_item.find('title')
        version_text = version_el.text if version_el is not None else "Unknown"

        description_el = latest_item.find('description')
        description_text = description_el.text if description_el is not None else ""

        pub_date_el = latest_item.find('pubDate')
        pub_date_text = pub_date_el.text if pub_date_el is not None else ""

        enclosure = latest_item.find('enclosure')
        download_url = enclosure.get('url') if enclosure is not None else ""
        length = enclosure.get('length') or "0"
        file_size = int(length) if length.isdigit() else 0

        return {
            'version': version_text,
            'description': description_text,
            'pub_date': pub_date_text,
            'download_url': download_url,
            'file_size': file_size
        }
    except Exception as e:
        print(f"Error parsing appcast: {e}")
        return None


def check_for_updates_async():
    """Check for updates in a separate thread"""
    try:
        current_version = "0.2.11"
        appcast_url = "https://raw.githubusercontent.com/Lixin-TU/AudioSpectroDemo/main/appcast.xml"

        update_info = parse_appcast_xml(appcast_url)
        if not update_info:
            update_checker.update_error.emit("Failed to fetch update information")
            return

        latest_version = update_info['version'].replace('Version ', '').strip()
        if latest_version != current_version:
            update_checker.update_available.emit(update_info)
        else:
            update_checker.update_not_available.emit(current_version)
    except Exception as e:
        update_checker.update_error.emit(f"Update check failed: {e}")


from dsp.mel import (
    wav_to_mel_image,
    TARGET_SR,
    MIN_FREQ,
    MAX_FREQ,
)

MEL_MAX = librosa.hz_to_mel(MAX_FREQ)
LUT = (plt.get_cmap("plasma")(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

def _hz_to_k_label(hz: int) -> str:
    return "0" if hz == 0 else f"{int(round(hz / 1000))}"

EXPORT_W = 800
EXPORT_H = 400

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AudioSpectroDemo")
        self._session_temp_files: list[str] = []
        self._session_temp_dirs: list[str] = []
        self.resize(900, 600)

        self._init_winsparkle()
        self._build_ui()

        update_checker.update_available.connect(self.on_update_available)
        update_checker.update_not_available.connect(self.on_update_not_available)
        update_checker.update_error.connect(self.on_update_error)

        self.open_btn.clicked.connect(self.open_wav)
        self.update_button.clicked.connect(self.download_update)
        QTimer.singleShot(0, self.check_for_updates)

    def _build_ui(self):
        central = pg.QtWidgets.QWidget(self)
        layout = QVBoxLayout(central)

        self.info_label = QLabel("UBCO-ISDPRL  •  AudioSpectroDemo")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)

        self.update_label = QLabel("Checking for updates...")
        self.update_label.setAlignment(Qt.AlignCenter)
        self.update_label.setStyleSheet("color: #666; font-size: 12px; padding: 5px;")
        layout.addWidget(self.update_label)

        self.update_button = QPushButton("Download Update")
        self.update_button.setVisible(False)
        layout.addWidget(self.update_button)

        self.open_btn = QPushButton("Open WAV files")
        self.export_checkbox = QCheckBox("Export spectrograms")
        layout.addWidget(self.open_btn)
        layout.addWidget(self.export_checkbox)
        layout.addStretch(1)

        self.setCentralWidget(central)

    def _init_winsparkle(self):
        global WINSPARKLE_LOG_PATH, _ws
        if platform.system() != "Windows":
            return
        try:
            candidates = [
                pathlib.Path(sys.executable).with_name("winsparkle.dll"),
                pathlib.Path(__file__).resolve().parent / "winsparkle.dll",
            ]
            dll_path = next((p for p in candidates if p.exists()), None)
            if not dll_path:
                return

            ctypes.WinDLL(str(dll_path))  # test load
            log_path = pathlib.Path(sys.executable).with_suffix(".winsparkle.log")
            WINSPARKLE_LOG_PATH = str(log_path)
            self._session_temp_files.append(WINSPARKLE_LOG_PATH)

            _ws = ctypes.WinDLL(str(dll_path))
            _ws.win_sparkle_set_appcast_url.argtypes = [ctypes.c_wchar_p]
            _ws.win_sparkle_set_app_details.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_wchar_p]
            _ws.win_sparkle_set_verbosity_level.argtypes = [ctypes.c_int]
            _ws.win_sparkle_set_log_path.argtypes = [ctypes.c_wchar_p]

            _ws.win_sparkle_set_appcast_url("https://raw.githubusercontent.com/Lixin-TU/AudioSpectroDemo/main/appcast.xml")
            _ws.win_sparkle_set_app_details("UBCO-ISDPRL", "AudioSpectroDemo", "0.2.11")
            _ws.win_sparkle_set_verbosity_level(2)
            _ws.win_sparkle_set_log_path(WINSPARKLE_LOG_PATH)
            _ws.win_sparkle_init()
        except OSError:
            _ws = None

    def check_for_updates(self):
        if platform.system() == "Windows" and _ws:
            try:
                _ws.win_sparkle_check_update_without_ui()
                threading.Thread(target=check_for_updates_async, daemon=True).start()
                QTimer.singleShot(2000, self._check_winsparkle_log)
            except Exception as e:
                self.on_update_error(f"WinSparkle check failed: {e}")
        else:
            threading.Thread(target=check_for_updates_async, daemon=True).start()

    def _check_winsparkle_log(self):
        if not WINSPARKLE_LOG_PATH or not os.path.exists(WINSPARKLE_LOG_PATH):
            return
        try:
            with open(WINSPARKLE_LOG_PATH, "r", encoding="utf-8") as f:
                last = f.readlines()[-1].strip()
                print(f"WinSparkle log: {last}")
        except Exception:
            pass

    def on_update_available(self, info):
        ver = info.get('version', 'Unknown')
        size_mb = info.get('file_size', 0) / (1024*1024)
        txt = f"🔄 New version available: {ver}"
        if size_mb > 0:
            txt += f" ({size_mb:.1f} MB)"
        self.update_label.setText(txt)
        self.update_label.setStyleSheet("color: #2196F3; font-weight: bold; font-size: 12px; padding: 5px;")
        self.update_button.setVisible(True)
        self.current_update_info = info

    def on_update_not_available(self, cur_ver):
        self.update_label.setText(f"✅ You're running the latest version ({cur_ver})")
        self.update_label.setStyleSheet("color: #4CAF50; font-size: 12px; padding: 5px;")

    def on_update_error(self, msg):
        self.update_label.setText(f"❌ Update check failed: {msg}")
        self.update_label.setStyleSheet("color: #F44336; font-size: 12px; padding: 5px;")

    def download_update(self):
        """Download update, show progress, remove old, and launch new."""
        info = getattr(self, 'current_update_info', {})
        url = info.get('download_url')
        if not url:
            return

        progress = QProgressDialog("Starting update...", None, 0, 3, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(False)
        progress.setValue(0)
        QApplication.processEvents()

        # Step 1: Download
        progress.setLabelText("Downloading update...")
        temp_dir = tempfile.mkdtemp(prefix="audiospec_update_")
        installer_name = os.path.basename(url)
        installer_path = os.path.join(temp_dir, installer_name)

        def _hook(count, block_size, total_size):
            pct = int(count * block_size * 100 / total_size) if total_size > 0 else 0
            progress.setValue(min(pct, 100))
            QApplication.processEvents()

        urllib.request.urlretrieve(url, installer_path, reporthook=_hook)
        progress.setValue(1)

        # Step 2: Prepare updater
        progress.setLabelText("Preparing updater...")
        QApplication.processEvents()
        old_exe = sys.executable
        bat_path = os.path.join(temp_dir, "apply_update.bat")
        with open(bat_path, "w", encoding="utf-8") as bat:
            bat.write(f"""@echo off
ping 127.0.0.1 -n 5 >nul
rd /s /q \"{os.path.dirname(old_exe)}\"
start "" \"{installer_path}\"
del "%~f0"
""" )
        progress.setValue(2)

        # Step 3: Launch new version
        progress.setLabelText("Launching new version...")
        QApplication.processEvents()
        subprocess.Popen(["cmd", "/c", bat_path], shell=False)
        progress.setValue(3)

        time.sleep(0.5)
        progress.close()
        QApplication.quit()

    def open_wav(self):
        dlg = QFileDialog(self)
        dlg.setFileMode(QFileDialog.ExistingFiles)
        dlg.setNameFilter("WAV (*.wav *.WAV)")
        if not dlg.exec():
            return
        files = dlg.selectedFiles()
        if not files:
            return

        progress = QProgressDialog("Processing audio...", None, 0, len(files), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.show()
        QApplication.processEvents()

        spectrograms = []
        for i, fp in enumerate(files):
            progress.setValue(i)
            QApplication.processEvents()
            img = wav_to_mel_image(fp)
            spectrograms.append((fp, img))
        progress.close()

        if self.export_checkbox.isChecked():
            for wav_path, img in spectrograms:
                norm = img - img.min()
                if norm.max() > 0:
                    norm = norm / norm.max()
                inds = (norm * 255).astype(np.uint8)
                rgb = LUT[inds]
                hop = 512
                export_dir = os.path.join(os.path.dirname(wav_path), "spectrograms")
                os.makedirs(export_dir, exist_ok=True)
                if export_dir not in self._session_temp_dirs:
                    self._session_temp_dirs.append(export_dir)
                duration_min = rgb.shape[1] * hop / TARGET_SR / 60.0

                fig = plt.figure(figsize=(EXPORT_W/300, EXPORT_H/300), dpi=300)
                ax = fig.add_subplot(111)
                fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12)
                ax.imshow(np.flipud(rgb), aspect='auto',extent=[0, duration_min, 0, MAX_FREQ], origin='lower')
                ax.set_xlabel("Time (min)",fontsize=3)
                ax.set_ylabel("Frequency (kHz)",fontsize=3)
                freqs=[0,2000,4000,6000,10000,12000,14000,16000]
                ax.set_yticks(freqs)
                ax.set_yticklabels([_hz_to_k_label(f) for f in freqs],fontsize=3)
                ax.set_ylim(0, MAX_FREQ)
                ax.tick_params(axis="both",which="both",direction="in",color="0.6",width=0.4,length=3,labelsize=3)
                for spine in ax.spines.values(): spine.set_linewidth(0.4); spine.set_color("0.6")
                ax.set_title(os.path.basename(wav_path),fontsize=5,pad=4)
                png = os.path.join(export_dir, os.path.splitext(os.path.basename(wav_path))[0] + ".png")
                fig.savefig(png,dpi=300,bbox_inches="tight",pad_inches=0.01)
                self._session_temp_files.append(png)
                plt.close(fig)

        # Viewer dialog (unchanged)
        dialog = QDialog(self)
        dialog.setWindowTitle("Spectrogram Viewer")
        main_layout = QVBoxLayout(dialog)
        pw = PlotWidget(background="w")
        main_layout.addWidget(pw)

        btn_layout = QHBoxLayout()
        prev_btn = QPushButton("Previous")
        next_btn = QPushButton("Next")
        btn_layout.addWidget(prev_btn); btn_layout.addWidget(next_btn)
        main_layout.addLayout(btn_layout)

        gain_layout = QHBoxLayout()
        gain_label = QLabel("Color gain (dB):")
        slider = QSlider(Qt.Horizontal); slider.setRange(0,40); slider.setValue(0)
        gain_value = QLabel("0 dB")
        gain_layout.addWidget(gain_label); gain_layout.addWidget(slider,1); gain_layout.addWidget(gain_value)
        main_layout.addLayout(gain_layout)

        small = self.font(); small.setPointSizeF(small.pointSizeF()*0.85)
        gain_label.setFont(small); gain_value.setFont(small)

        from PySide6.QtGui import QTransform
        index=0
        def update(idx):
            pw.clear()
            fp,img = spectrograms[idx]
            db=slider.value(); img_f=img.astype(np.float32)/255.0
            img_f=np.clip(img_f*(10**(db/20.0)),0,1); disp=(img_f*255).astype(np.uint8)
            dialog.setWindowTitle(os.path.basename(fp))
            freqs=[0,2000,4000,6000,10000,12000,14000,16000]
            ticks=[(f,_hz_to_k_label(f)) for f in freqs]
            hop=512; scale_x=hop/TARGET_SR/60
            pw.setLabel("bottom","Time",units="min"); pw.setLabel("left","Freq",units="Hz")
            pw.getAxis("left").setTicks([ticks])
            item=ImageItem(disp)
            item.setLookupTable(LUT,update=True)
            item.setOpts(axisOrder="row-major")
            item.setTransform(QTransform().scale(scale_x,-MAX_FREQ/(img.shape[0]-1)))
            item.setPos(0,MAX_FREQ); pw.addItem(item); pw.setLimits(yMin=0,yMax=MAX_FREQ)
            prev_btn.setEnabled(idx>0); next_btn.setEnabled(idx<len(spectrograms)-1)
        slider.valueChanged.connect(lambda v: (gain_value.setText(f"{int(v)} dB"), update(index)))
        prev_btn.clicked.connect(lambda: update(index-1))
        next_btn.clicked.connect(lambda: update(index+1))
        update(0)
        dialog.resize(EXPORT_W, EXPORT_H+120)
        dialog.exec()

    def closeEvent(self,event):
        global _ws
        if platform.system()=="Windows" and _ws:
            try: _ws.win_sparkle_cleanup()
            except: pass
        for p in self._session_temp_files:
            try: os.remove(p)
            except: pass
        for d in self._session_temp_dirs:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except:
                pass
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec())
