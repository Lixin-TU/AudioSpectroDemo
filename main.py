import sys
import time
import os
import platform
import pathlib
import logging  # Add this import
import tempfile  # Add this missing import
import urllib.request  # Add this missing import
import subprocess  # Add this missing import

import ctypes
import shutil
import re

# Set up logging to file
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "update_log.txt")
if getattr(sys, 'frozen', False):
    # If running as compiled exe
    app_dir = os.path.dirname(sys.executable)
    log_path = os.path.join(app_dir, "update_log.txt")

# Logging to file disabled to prevent update_log.txt creation
# logging.basicConfig(
#     filename=log_path,
#     level=logging.DEBUG,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )

# --- Compatibility shims for the frozen Windows build -----------------------
# Ensure 'unittest' is always importable (PyInstaller one‑file may omit it)
try:
    import unittest  # noqa: F401
except ModuleNotFoundError:
    import types as _types, sys as _sys
    _stub = _types.ModuleType("unittest")
    _sys.modules["unittest"] = _stub

# Hide the temporary PyInstaller extraction folder (e.g. _MEI12345) and
# remove any *older* _MEI folders so only the latest one remains.
if hasattr(sys, "_MEIPASS") and platform.system() == "Windows":
    try:
        mei_dir = pathlib.Path(sys._MEIPASS).resolve()
        # Delete all sibling _MEI* directories except the one in use
        for p in mei_dir.parent.glob("_MEI*"):
            if p.is_dir() and p != mei_dir:
                shutil.rmtree(p, ignore_errors=True)
        # Finally hide the current extraction folder
        FILE_ATTRIBUTE_HIDDEN = 0x02
        ctypes.windll.kernel32.SetFileAttributesW(str(mei_dir), FILE_ATTRIBUTE_HIDDEN)
    except Exception:
        pass
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import librosa
import threading
from dsp.update import UpdateChecker, update_checker, parse_appcast_xml, filename_from_url, check_for_updates_async

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
    QMessageBox,
    QWidget,
    QProgressBar,
    QFrame,
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QThread, QRectF, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPalette

import pyqtgraph as pg
from pyqtgraph import PlotWidget, ImageItem

# Add matplotlib imports for the new viewer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle  # Add this import

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

# Audio processing worker thread
class AudioProcessor(QThread):
    progress_updated = Signal(int)
    processing_complete = Signal(list)
    processing_error = Signal(str)
    
    def __init__(self, files):
        super().__init__()
        self.files = files
        
    def run(self):
        try:
            from dsp.mel import wav_to_mel_image
            spectrograms = []
            for i, fp in enumerate(self.files):
                self.progress_updated.emit(i)
                img = wav_to_mel_image(fp)
                spectrograms.append((fp, img))
            self.processing_complete.emit(spectrograms)
        except Exception as e:
            self.processing_error.emit(str(e))

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

# Convert dimensions: H=6cm, W=17cm at 300 DPI for export only
# 1 inch = 2.54 cm, so: cm * 300 DPI / 2.54 = pixels
EXPORT_W = int(17 * 300 / 2.54)  # 17cm = ~2008 pixels at 300 DPI
EXPORT_H = int(6 * 300 / 2.54)   # 6cm = ~709 pixels at 300 DPI

# Original dimensions for viewer dialog
VIEWER_W = 800
VIEWER_H = 400

class LoadingOverlay(QWidget):
    """Loading overlay widget that appears over the viewer during processing"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("LoadingOverlay")
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        
        # Set up semi-transparent background
        self.setStyleSheet("""
            QWidget#LoadingOverlay {
                background-color: rgba(255, 255, 255, 220);
                border-radius: 10px;
            }
        """)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        
        # Loading message
        self.message_label = QLabel("Loading audio...")
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #333;
                margin-bottom: 20px;
                padding: 10px;
                background-color: rgba(255, 255, 255, 100);
                border-radius: 5px;
            }
        """)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(350)
        self.progress_bar.setFixedHeight(25)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #4CAF50;
                border-radius: 12px;
                text-align: center;
                font-size: 14px;
                font-weight: bold;
                background-color: rgba(255, 255, 255, 200);
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2=0,
                    stop:0 #4CAF50, stop:0.5 #66BB6A, stop:1 #45a049);
                border-radius: 10px;
                margin: 1px;
            }
        """)
        
        # Percentage label
        self.percentage_label = QLabel("0%")
        self.percentage_label.setAlignment(Qt.AlignCenter)
        self.percentage_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #333;
                margin-top: 15px;
                padding: 5px;
                background-color: rgba(255, 255, 255, 100);
                border-radius: 5px;
            }
        """)
        
        layout.addWidget(self.message_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.percentage_label)
        
        # Initially hidden
        self.hide()
    
    def show_loading(self, message="Loading audio..."):
        """Show the loading overlay with a message"""
        self.message_label.setText(message)
        self.progress_bar.setValue(0)
        self.percentage_label.setText("0%")
        self.show()
        self.raise_()
        # Force immediate repaint
        QApplication.processEvents()
    
    def update_progress(self, value, message=None):
        """Update the progress bar and optionally the message"""
        self.progress_bar.setValue(value)
        self.percentage_label.setText(f"{value}%")
        if message:
            self.message_label.setText(message)
        # Force immediate update
        QApplication.processEvents()
    
    def hide_loading(self):
        """Hide the loading overlay"""
        self.hide()
        QApplication.processEvents()
    
    def resizeEvent(self, event):
        """Resize to cover the entire parent widget"""
        if self.parent():
            self.resize(self.parent().size())
        super().resizeEvent(event)

class AudioCache:
    """Cache for processed audio data to speed up navigation"""
    
    def __init__(self, max_size=10):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []  # Track access order for LRU eviction
    
    def get(self, file_path):
        """Get cached audio data for a file"""
        if file_path in self.cache:
            # Move to end of access order (most recently used)
            self.access_order.remove(file_path)
            self.access_order.append(file_path)
            return self.cache[file_path]
        return None
    
    def put(self, file_path, data):
        """Store audio data in cache"""
        # Remove if already exists
        if file_path in self.cache:
            self.access_order.remove(file_path)
        
        # Evict least recently used if cache is full
        elif len(self.cache) >= self.max_size:
            lru_file = self.access_order.pop(0)
            del self.cache[lru_file]
        
        self.cache[file_path] = data
        self.access_order.append(file_path)
    
    def clear(self):
        """Clear all cached data"""
        self.cache.clear()
        self.access_order.clear()

class AudioProcessorWithProgress(QThread):
    """Enhanced audio processor with detailed progress reporting"""
    progress_updated = Signal(int, str)  # progress value, message
    processing_complete = Signal(object)  # processed data
    processing_error = Signal(str)
    
    def __init__(self, file_path, cache=None):
        super().__init__()
        self.file_path = file_path
        self.cache = cache
        self._is_cancelled = False
        
    def cancel(self):
        """Cancel the processing"""
        self._is_cancelled = True
        
    def run(self):
        try:
            # Check cache first
            if self.cache:
                cached_data = self.cache.get(self.file_path)
                if cached_data:
                    # Even for cached data, show brief loading for user feedback
                    self.progress_updated.emit(20, "Loading from cache...")
                    time.sleep(0.1)  # Brief delay to show progress
                    self.progress_updated.emit(100, "Complete!")
                    time.sleep(0.1)
                    self.processing_complete.emit(cached_data)
                    return
            
            if self._is_cancelled:
                return
                
            # Step 1: Reading file
            self.progress_updated.emit(10, "Reading audio file...")
            time.sleep(0.1)  # Slightly longer to ensure visibility
            
            if self._is_cancelled:
                return
                
            y, sr = librosa.load(self.file_path, sr=TARGET_SR, mono=True)
            
            # Step 2: Processing audio
            self.progress_updated.emit(30, "Processing audio data...")
            time.sleep(0.1)
            
            if self._is_cancelled:
                return
            
            # Step 3: Generating spectrogram
            self.progress_updated.emit(50, "Generating mel spectrogram...")
            
            n_mels = 256
            hop_length = 512
            n_fft = 2048
            
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr, 
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                fmin=MIN_FREQ,
                fmax=MAX_FREQ
            )
            
            if self._is_cancelled:
                return
            
            # Step 4: Converting to dB
            self.progress_updated.emit(75, "Converting to dB scale...")
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            if self._is_cancelled:
                return
            
            # Step 5: Finalizing
            self.progress_updated.emit(90, "Finalizing visualization...")
            time.sleep(0.1)
            
            duration_min = len(y) / sr / 60.0
            
            result = {
                'file_path': self.file_path,
                'audio_data': y,
                'sample_rate': sr,
                'spectrogram': mel_spec_db,
                'duration_min': duration_min
            }
            
            # Cache the result for future use
            if self.cache:
                self.cache.put(self.file_path, result)
            
            self.progress_updated.emit(100, "Complete!")
            time.sleep(0.1)  # Brief pause to show completion
            self.processing_complete.emit(result)
            
        except Exception as e:
            if not self._is_cancelled:
                self.processing_error.emit(str(e))

# --- Main Window -----------------------------------------------------------
class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AudioSpectro")
        self._session_temp_files: list[str] = []
        self.resize(900, 600)
        self.audio_processor = None
        self._download_cancelled = False
        
        # Initialize audio cache
        self.audio_cache = AudioCache(max_size=15)  # Cache up to 15 files

        self._init_winsparkle()
        self._build_ui()
        self._cleanup_old_versions()

        update_checker.update_available.connect(self.on_update_available)
        update_checker.update_not_available.connect(self.on_update_not_available)
        update_checker.update_error.connect(self.on_update_error)

        self.open_btn.clicked.connect(self.open_wav)
        self.update_button.clicked.connect(self.download_update)
        QTimer.singleShot(0, self.check_for_updates)

    def _cleanup_old_versions(self):
        """
        Delete any other AudioSpectroDemo‑vX.Y.Z.exe files in the app folder
        except the one currently running.
        """
        info = self._get_current_app_info()
        exe_path = info['exe_path'].resolve()
        app_dir  = info['app_dir']
        pat = re.compile(r"AudioSpectroDemo-v[\d.]+\.exe$", re.IGNORECASE)
        for p in app_dir.glob("AudioSpectroDemo-v*.exe"):
            try:
                if p.resolve() == exe_path:
                    continue
                if pat.search(p.name):
                    p.unlink(missing_ok=True)
            except Exception:
                pass

    def _build_ui(self):
        central = pg.QtWidgets.QWidget(self)
        layout = QVBoxLayout(central)

        self.info_label = QLabel("UBCO-ISDPRL  •  AudioSpectro")
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
        self.export_checkbox = QCheckBox("Export spectrograms (+10 dB)")
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
            _ws.win_sparkle_set_app_details("UBCO-ISDPRL", "AudioSpectroDemo", "0.2.30")
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

    def _get_current_app_info(self):
        """Get information about the current running application"""
        if getattr(sys, "frozen", False):
            # Running as PyInstaller executable
            current_exe = pathlib.Path(sys.executable).resolve()
            app_dir = current_exe.parent
            return {
                'exe_path': current_exe,
                'app_dir': app_dir,
                'is_frozen': True,
                'exe_name': current_exe.name
            }
        else:
            # Running as script
            script_path = pathlib.Path(__file__).resolve()
            return {
                'exe_path': script_path,
                'app_dir': script_path.parent,
                'is_frozen': False,
                'exe_name': script_path.name
            }
    
    def _create_update_script(self, new_exe_path, current_app_info, final_exe_name):
        """Create a batch script to move the new EXE into place and launch it (DEBUG VERSION: writes to Desktop)."""
        import os
        current_exe = current_app_info['exe_path']
        app_dir = current_app_info['app_dir']
        new_exe_name = final_exe_name
        final_new_exe_path = os.path.join(str(app_dir), new_exe_name)
        script_content = f"""@echo off
REM AudioSpectroDemo Auto-Updater (Batch DEBUG)

setlocal ENABLEDELAYEDEXPANSION
set "SRC={new_exe_path}"
set "DST={final_new_exe_path}"
set "OLD={current_exe}"

echo SRC=!SRC!
echo DST=!DST!
echo OLD=!OLD!

REM Wait for the old EXE to exit and unlock (max 30 attempts, 30*0.7s)
for /L %%i in (1,1,30) do (
    ping 127.0.0.1 -n 2 >nul
    ren "!OLD!" "!OLD!.tmp" >nul 2>&1 && goto moved
)
echo ERROR: Old EXE did not exit in 30s
pause
exit /b 1
:moved
move /Y "!SRC!" "!DST!" >nul
del /F /Q "!OLD!.tmp" >nul 2>&1
start "" "!DST!"
pause
del "%~f0"
exit /b 0
"""
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        batch_path = os.path.join(desktop, "audiospectro_updater.bat")
        with open(batch_path, "w", encoding="utf-8") as script_file:
            script_file.write(script_content)
        return batch_path, 'batch'

    def download_update(self):
        """Download and install update - replaces old version with new versioned executable"""
        info = getattr(self, 'current_update_info', {})
        url = info.get('download_url')
        if not url:
            return

        # Extract version info for display
        new_version = info.get('version', 'Unknown').replace('Version ', '').strip()
        file_size_mb = info.get('file_size', 0) / (1024*1024)
        size_text = f" ({file_size_mb:.1f} MB)" if file_size_mb > 0 else ""

        # Get current app info to show current version
        current_app_info = self._get_current_app_info()
        current_name = current_app_info['exe_path'].name
        new_exe_name = filename_from_url(url)

        reply = QMessageBox.question(
            self,
            "Update Confirmation",
            f"This will update to {new_exe_name}{size_text}.\n\n"
            "The update process will:\n"
            "1. Download the new version\n"
            "2. Back up the current version\n"
            "3. Replace the old executable with the new one\n"
            "4. Launch the new version automatically\n"
            "5. Remove the old version\n\n"
            "Do you want to continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        if reply != QMessageBox.Yes:
            return

        # Create path for new executable in temp directory
        if not new_exe_name:
            new_exe_name = f"AudioSpectroDemo-v{new_version}.exe"

        # Use a safe suffix
        safe_suffix = "_" + new_exe_name.replace("?", "_")
        temp_new_exe = tempfile.NamedTemporaryFile(
            suffix=safe_suffix,
            delete=False
        )
        temp_new_exe.close()
        new_exe_temp_path = temp_new_exe.name

        # IMPORTANT: Reset the download cancelled flag before starting
        self._download_cancelled = False
        
        # Create a progress dialog with a more controlled cancel behavior
        progress = QProgressDialog("Downloading update...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.setValue(0)
        
        # Store progress dialog reference to access it later
        self._download_progress = progress
        self._download_completed = False  # New flag to track download completion
        
        # Only connect once to avoid duplicate signals
        progress.canceled.connect(self._handle_download_cancel)

        def _safely_close_progress():
            if hasattr(self, '_download_progress') and self._download_progress:
                try:
                    # Block signals before closing to prevent unwanted cancel signals
                    self._download_progress.blockSignals(True)
                    self._download_progress.close()
                    self._download_progress = None
                    logging.debug("Progress dialog closed safely")
                except Exception as e:
                    pass
        
        def _hook(count, block_size, total_size):
            if self._download_cancelled:
                raise Exception("Download cancelled by user")
            if total_size > 0:
                pct = int(count * block_size * 100 / total_size)
                if hasattr(self, '_download_progress') and self._download_progress:
                    self._download_progress.setValue(min(pct, 100))
                    downloaded_mb = (count * block_size) / (1024*1024)
                    total_mb = total_size / (1024*1024)
                    self._download_progress.setLabelText(f"Downloading {new_exe_name}... {downloaded_mb:.1f}/{total_mb:.1f} MB")
                QApplication.processEvents()

        try:
            # Add SSL context for compiled executable
            import ssl
            ctx = ssl._create_unverified_context()
            
            # Add headers to avoid potential blocking
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'AudioSpectroDemo/0.2.30')

            print(f"Starting download from: {url}")  # Debug output
            print(f"Downloading to: {new_exe_temp_path}")  # Debug output
            
            # Use urlopen with SSL context instead of urlretrieve
            with urllib.request.urlopen(req, context=ctx) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                block_size = 8192
                
                with open(new_exe_temp_path, 'wb') as f:
                    while True:
                        if self._download_cancelled:
                            raise Exception("Download cancelled by user")
                            
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                            
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Update progress
                        if total_size > 0:
                            pct = int(downloaded * 100 / total_size)
                            if hasattr(self, '_download_progress') and self._download_progress:
                                self._download_progress.setValue(min(pct, 100))
                                downloaded_mb = downloaded / (1024*1024)
                                total_mb = total_size / (1024*1024)
                                self._download_progress.setLabelText(f"Downloading {new_exe_name}... {downloaded_mb:.1f}/{total_mb:.1f} MB")
                            QApplication.processEvents()

            print("Download completed successfully")  # Debug output
            
            # Ensure progress dialog is closed - safely
            _safely_close_progress()
            
            # Force UI update
            QApplication.processEvents()

            # Only proceed with update if download was not cancelled
            if self._download_cancelled:
                return
            
            # Additional safeguard: verify the download worked and file exists
            if not os.path.exists(new_exe_temp_path) or os.path.getsize(new_exe_temp_path) == 0:
                QMessageBox.critical(self, "Update Error", "Downloaded file is missing or empty.")
                return

            # Verify downloaded file
            file_size = os.path.getsize(new_exe_temp_path)
            if file_size == 0:
                raise Exception("Downloaded file is empty")
            elif file_size < 1000000:  # Less than 1MB seems to small for this app
                raise Exception(f"Downloaded file seems to small ({file_size} bytes)")

            # If running from source (not frozen), we cannot auto‑replace the script.
            if not current_app_info['is_frozen']:
                QMessageBox.information(
                    self,
                    "Update downloaded",
                    "The update was downloaded to:\n"
                    f"{new_exe_temp_path}\n\n"
                    "Because you're running the Python source in VS Code, "
                    "the auto‑update step is skipped.\n"
                    "Run the packaged EXE to enable one‑click updates."
                )
                return

            # Show success message with proper parent and modality
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Update ready")
            msg_box.setText("The new version has been downloaded.\nAudioSpectroDemo will now restart.")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.setModal(True)
            result = msg_box.exec()

            # Move (or overwrite) into the app folder
            final_path = os.path.join(str(current_app_info['app_dir']), new_exe_name)
            try:
                if os.path.exists(final_path):
                    pass
                # First check permissions
                access_check = os.access(os.path.dirname(final_path), os.W_OK)
                try:
                    shutil.move(new_exe_temp_path, final_path)
                except PermissionError:
                    shutil.copy2(new_exe_temp_path, final_path)
                    os.remove(new_exe_temp_path)
                # Verify the file was actually moved
                if os.path.exists(final_path):
                    pass
                else:
                    pass
            except Exception as e:
                error_msg = f"Failed to swap executable:\n{e}"
                QMessageBox.critical(self, "Update Error", error_msg)
                try:
                    os.remove(new_exe_temp_path)
                except Exception as e2:
                    pass
                return

            # Launch the fresh version
            try:
                new_process = subprocess.Popen([str(final_path)])
                # Remove old executable after launching new process
                try:
                    old_exe = current_app_info['exe_path']
                    if os.path.exists(old_exe):
                        os.remove(old_exe)
                except Exception as e:
                    pass
            except Exception as e:
                QMessageBox.critical(self, "Launch Error", f"Failed to start new version: {e}")
                return

            # Clean up and exit this (old) process
            self._prepare_for_update()
            self._cleanup_old_versions()
            os._exit(0)

        except Exception as e:
            # Ensure progress dialog is closed on error
            if progress:
                progress.close()
                progress = None
            
            # Force UI update
            QApplication.processEvents()
            
            if not self._download_cancelled:
                error_msg = str(e)
                error_type = type(e).__name__
                
                # More detailed error information
                detailed_msg = f"Error Type: {error_type}\nError: {error_msg}"
                
                if "HTTP Error" in error_msg:
                    detailed_msg += "\n\nThis might be a temporary network issue. Please try again later."
                elif "SSL" in error_msg or "certificate" in error_msg.lower():
                    detailed_msg += "\n\nSSL certificate issue. This is common in compiled executables."
                elif "urlopen error" in error_msg:
                    detailed_msg += "\n\nNetwork connection issue. Check your internet connection."
                elif "Permission" in error_msg:
                    detailed_msg += "\n\nPermission denied. Try running as administrator."
                    
                QMessageBox.critical(self, "Update Error", f"Failed to download update:\n{detailed_msg}")
            
            try:
                os.remove(new_exe_temp_path)
            except Exception as cleanup_error:
                print(f"Cleanup error: {cleanup_error}")

    def _cancel_download(self, file_path):
        """Handle download cancellation - legacy method"""
        self._download_cancelled = True
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:  # Fix: add 'as' keyword
            pass

    def _handle_download_cancel(self):
        """Handle download cancellation from the progress dialog"""
        self._download_cancelled = True
        if hasattr(self, '_download_progress') and self._download_progress:
            self._download_progress.close()
            self._download_progress = None

    def _on_download_cancelled(self):
        """Handle download cancellation from the progress dialog"""
        # Only handle cancellation if download hasn't completed
        if not hasattr(self, '_download_completed') or not self._download_completed:
            self._download_cancelled = True
            if hasattr(self, '_download_progress') and self._download_progress:
                self._download_progress.close()
                self._download_progress = None

    def _launch_update_process(self, new_exe_temp_path, current_app_info):
        """Launch the update process using a script"""
        dest_exe_name = filename_from_url(
            self.current_update_info.get("download_url", "")
        )
        try:
            # Create update script
            script_path, script_type = self._create_update_script(
                new_exe_temp_path, current_app_info, dest_exe_name
            )
            final_path = current_app_info['app_dir'] / dest_exe_name

            debug_msg = (
                f"About to run updater.\n"
                f"Script: {script_path}\n"
                f"Temp new exe: {new_exe_temp_path}\n"
                f"Current exe: {current_app_info['exe_path']}\n"
                f"Target exe: {final_path}\n"
            )
            QMessageBox.information(
                self,
                "DEBUG",
                debug_msg
            )

            # Prepare for shutdown
            self._prepare_for_update()

            if script_type == 'batch':
                # Use /k for debugging (cmd window stays open)
                import subprocess
                subprocess.Popen([
                    'cmd', '/k', script_path
                ])
                os._exit(0)
            elif script_type == 'powershell':
                subprocess.Popen([
                    'powershell',
                    '-WindowStyle', 'Hidden',
                    '-ExecutionPolicy', 'Bypass',
                    '-File', script_path
                ], creationflags=subprocess.CREATE_NO_WINDOW)
                os._exit(0)
            else:
                subprocess.Popen(['/bin/bash', script_path])
                os._exit(0)

        except Exception as e:
            error_msg = f"Failed to launch update process: {e}"
            print(error_msg)
            QMessageBox.critical(self, "Update Error", error_msg)

    def _prepare_for_update(self):
        """Prepare the application for update by cleaning up resources"""
        # Cancel any running audio processing
        if self.audio_processor and self.audio_processor.isRunning():
            self.audio_processor.terminate()
            self.audio_processor.wait()
            
        # Cleanup WinSparkle
        global _ws
        if platform.system() == "Windows" and _ws:
            try:
                _ws.win_sparkle_cleanup()
            except:
                pass

    def open_wav(self):
        dlg = QFileDialog(self)
        dlg.setFileMode(QFileDialog.ExistingFiles)
        dlg.setNameFilter("WAV (*.wav *.WAV)")
        if not dlg.exec():
            return
        files = dlg.selectedFiles()
        if not files:
            return

        # Disable UI during processing
        self.open_btn.setEnabled(False)
        self.export_checkbox.setEnabled(False)

        # Create progress dialog
        self.progress = QProgressDialog("Processing audio...", "Cancel", 0, len(files), self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.show()
        QApplication.processEvents()

        # Start audio processing in separate thread
        self.audio_processor = AudioProcessor(files)
        self.audio_processor.progress_updated.connect(self.on_audio_progress)
        self.audio_processor.processing_complete.connect(self.on_audio_complete)
        self.audio_processor.processing_error.connect(self.on_audio_error)
        self.progress.canceled.connect(self.cancel_processing)
        self.audio_processor.start()

    def on_audio_progress(self, value):
        if hasattr(self, 'progress'):
            self.progress.setValue(value)
            QApplication.processEvents()

    def cancel_processing(self):
        if self.audio_processor and self.audio_processor.isRunning():
            self.audio_processor.terminate()
            self.audio_processor.wait()
        self.cleanup_processing()

    def cleanup_processing(self):
        # Re-enable UI
        self.open_btn.setEnabled(True)
        self.export_checkbox.setEnabled(True)
        prog = getattr(self, 'progress', None)
        if prog is not None:
            prog.close()
            # keep the attribute but clear it to avoid double‑deletion issues
            self.progress = None

    def on_audio_error(self, error_msg):
        self.cleanup_processing()
        QMessageBox.critical(self, "Processing Error", f"Failed to process audio: {error_msg}")

    def on_audio_complete(self, spectrograms):
        self.cleanup_processing()
        
        if self.export_checkbox.isChecked():
            self.export_spectrograms(spectrograms)

        # Show viewer dialog
        self.show_spectrogram_viewer(spectrograms)

    def export_spectrograms(self, spectrograms):
        """Export spectrograms with waveforms to PNG files"""
        for wav_path, img in spectrograms:
            # Load audio for waveform
            y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
            
            # Apply 10 dB amplification to waveform (allowing clipping)
            gain_factor = 10 ** (10 / 20)  # 10 dB gain
            y_amplified = y * gain_factor
            # Clip to [-1, 1] range to allow clipping
            y_amplified = np.clip(y_amplified, -1.0, 1.0)
            
            norm = img - img.min()
            if norm.max() > 0:
                norm = norm / norm.max()
            inds = (norm * 255).astype(np.uint8)
            rgb = LUT[inds]
            hop = 512
            export_dir = os.path.join(os.path.dirname(wav_path), "spectrograms")
            os.makedirs(export_dir, exist_ok=True)
            duration_min = rgb.shape[1] * hop / TARGET_SR / 60.0

            # Create figure with two subplots (waveform on top, spectrogram on bottom)
            fig = plt.figure(figsize=(EXPORT_W/300, EXPORT_H/300), dpi=300)
            
            # Waveform subplot (top 30% of the figure)
            ax1 = fig.add_subplot(2, 1, 1)
            time_axis = np.linspace(0, duration_min, len(y_amplified))
            ax1.plot(time_axis, y_amplified, color='black', linewidth=0.3)
            ax1.set_ylabel("Amplitude", fontsize=3)
            ax1.set_xlim(0, duration_min)  # Same x-scale as spectrogram
            ax1.tick_params(axis="both", which="both", direction="in", color="0.6", width=0.4, length=3, labelsize=3)
            ax1.set_xticklabels([])  # Remove x-axis labels from top plot
            for spine in ax1.spines.values(): spine.set_linewidth(0.4); spine.set_color("0.6")
            
            # Spectrogram subplot (bottom 70% of the figure)
            ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)  # Share x-axis with waveform
            # Flip the spectrogram vertically for correct orientation with low frequencies at bottom
            ax2.imshow(np.flipud(rgb), aspect='auto', extent=[0, duration_min, 0, MAX_FREQ], origin='upper')
            ax2.set_xlabel("Time (min)", fontsize=3)
            ax2.set_ylabel("Frequency (kHz)", fontsize=3)
            freqs=[0,2000,4000,6000,10000,12000,14000,16000]
            ax2.set_yticks(freqs)
            ax2.set_yticklabels([_hz_to_k_label(f) for f in freqs], fontsize=3)
            ax2.set_ylim(0, MAX_FREQ)
            ax2.set_xlim(0, duration_min)  # Explicitly set same x-scale
            ax2.tick_params(axis="both", which="both", direction="in", color="0.6", width=0.4, length=3, labelsize=3)
            # Ensure x-axis labels are visible on the bottom plot
            ax2.tick_params(axis='x', labelbottom=True)
            for spine in ax2.spines.values(): spine.set_linewidth(0.4); spine.set_color("0.6")
            
            # Adjust layout to minimize space between subplots
            fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12, hspace=0.25)  # Reduced hspace
            fig.suptitle(os.path.basename(wav_path), csize=3, y=0.95)
            
            png = os.path.join(export_dir, os.path.splitext(os.path.basename(wav_path))[0] + ".png")
            # The file will be automatically overwritten if it exists (default behavior)
            fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.01)
            plt.close(fig)

    def show_spectrogram_viewer(self, spectrograms):
        """Show the waveform and spectrogram viewer dialog using matplotlib with loading bar"""
        
        # Show viewer window immediately with loading state
        dialog = QDialog(self)
        dialog.setWindowTitle("Waveform & Spectrogram Viewer - Loading...")
        dialog.resize(1000, 600)
        main_layout = QVBoxLayout(dialog)
        
        # Create matplotlib figure and canvas first
        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(self.canvas)
        
        # Create loading overlay that covers the entire canvas
        self.loading_overlay = LoadingOverlay(canvas_container)
        
        main_layout.addWidget(canvas_container)

        # Create navigation controls
        btn_layout = QHBoxLayout()

        prev_btn = QPushButton("Previous")
        prev_btn.setEnabled(False)
        prev_btn.setFixedWidth(80)

        analysis_btn = QPushButton("Analysis")
        analysis_btn.setFixedSize(60, 60)
        analysis_btn.setStyleSheet(
            "QPushButton {"
            "    background-color: #FF7777;"
            "    color: white;"
            "    border: 2px solid white;"
            "    border-radius: 30px;"
            "    font-weight: bold;"
            "    font-size: 9px;"
            "}"
            "QPushButton:hover {"
            "    background-color: #FF9999;"
            "}"
            "QPushButton:pressed {"
            "    background-color: #FF5555;"
            "}"
        )

        next_btn = QPushButton("Next")
        next_btn.setEnabled(len(spectrograms) > 1)
        next_btn.setFixedWidth(80)

        btn_layout.addWidget(prev_btn)
        btn_layout.addStretch(2)
        btn_layout.addWidget(analysis_btn)
        btn_layout.addStretch(2)
        btn_layout.addWidget(next_btn)

        main_layout.addLayout(btn_layout)

        # Analysis options (same as before)
        analysis_options_layout = QHBoxLayout()
        
        remove_pulse_btn = QPushButton("Remove Pulse")
        remove_pulse_btn.setFixedSize(100, 30)
        remove_pulse_btn.setStyleSheet(
            "QPushButton {"
            "    background-color: #4CAF50;"
            "    color: white;"
            "    border: 1px solid white;"
            "    border-radius: 15px;"
            "    font-size: 8px;"
            "    font-weight: bold;"
            "}"
            "QPushButton:hover {"
            "    background-color: #66BB6A;"
            "}"
            "QPushButton:pressed {"
            "    background-color: #388E3C;"
            "}"
        )
        
        anomaly_detection_btn = QPushButton("Anomaly Detection")
        anomaly_detection_btn.setFixedSize(120, 30)
        anomaly_detection_btn.setEnabled(True)  # Enable the button
        anomaly_detection_btn.setStyleSheet(
            "QPushButton {"
            "    background-color: #2196F3;"  # Blue color for active button
            "    color: white;"
            "    border: 1px solid white;"
            "    border-radius: 15px;"
            "    font-size: 8px;"
            "    font-weight: bold;"
            "}"
            "QPushButton:hover {"
            "    background-color: #42A5F5;"
            "}"
            "QPushButton:pressed {"
            "    background-color: #1976D2;"
            "}"
        )
        
        ai_anomaly_btn = QPushButton("AI Anomaly Detection")
        ai_anomaly_btn.setFixedSize(140, 30)
        ai_anomaly_btn.setEnabled(False)
        ai_anomaly_btn.setStyleSheet(
            "QPushButton {"
            "    background-color: #CCCCCC;"
            "    color: #666666;"
            "    border: 1px solid #999999;"
            "    border-radius: 15px;"
            "    font-size: 8px;"
            "}"
        )
        
        remove_pulse_btn.setVisible(False)
        anomaly_detection_btn.setVisible(False)
        ai_anomaly_btn.setVisible(False)
        
        analysis_options_layout.addStretch(1)
        analysis_options_layout.addWidget(remove_pulse_btn)
        analysis_options_layout.addWidget(anomaly_detection_btn)
        analysis_options_layout.addWidget(ai_anomaly_btn)
        analysis_options_layout.addStretch(1)
        
        main_layout.addLayout(analysis_options_layout)
        
        options_visible = False

        def toggle_analysis_options():
            nonlocal options_visible
            options_visible = not options_visible
            
            if options_visible:
                remove_pulse_btn.setVisible(True)
                anomaly_detection_btn.setVisible(True)
                ai_anomaly_btn.setVisible(True)
                analysis_btn.setText("Close")
            else:
                remove_pulse_btn.setVisible(False)
                anomaly_detection_btn.setVisible(False)
                ai_anomaly_btn.setVisible(False)
                analysis_btn.setText("Analysis")

        analysis_btn.clicked.connect(toggle_analysis_options)

        def run_remove_pulse(spectrogram_data):
            try:
                from dsp.remove_pulse import process_remove_pulse
                fp, img = spectrogram_data
                print(f"Running Harmonic Extraction on: {os.path.basename(fp)}")
                
                # Show processing overlay
                self.loading_overlay.show_loading("Processing harmonic extraction...")
                QApplication.processEvents()
                
                # Process the audio
                result = process_remove_pulse(fp)
                
                if result['status'] == 'success':
                    # Update viewer with processed audio
                    processed_data = {
                        'file_path': fp,
                        'audio_data': result['processed_audio'],
                        'sample_rate': result['sample_rate'],
                        'spectrogram': None,  # Will be generated in update_viewer_display
                        'duration_min': len(result['processed_audio']) / result['sample_rate'] / 60.0,
                        'is_processed': True,
                        'processing_info': {
                            'method': 'Harmonic Extraction'
                        }
                    }
                    
                    # Generate new spectrogram for processed audio
                    y_processed = result['processed_audio']
                    sr = result['sample_rate']
                    
                    self.loading_overlay.update_progress(50, "Generating processed spectrogram...")
                    QApplication.processEvents()
                    
                    n_mels = 256
                    hop_length = 512
                    n_fft = 2048
                    
                    mel_spec = librosa.feature.melspectrogram(
                        y=y_processed, 
                        sr=sr, 
                        n_mels=n_mels,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        fmin=MIN_FREQ,
                        fmax=MAX_FREQ
                    )
                    
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    processed_data['spectrogram'] = mel_spec_db
                    
                    self.loading_overlay.update_progress(90, "Finalizing...")
                    QApplication.processEvents()
                    
                    # Update the display with processed data
                    update_viewer_display(processed_data)
                    
                    self.loading_overlay.hide_loading()
                    
                    # Show processing results
                    info_msg = (
                        f"Harmonic extraction completed!\n\n"
                        f"The viewer now shows the processed audio with harmonic components only."
                    )
                    QMessageBox.information(dialog, "Harmonic Extraction Complete", info_msg)
                else:
                    self.loading_overlay.hide_loading()
                    QMessageBox.critical(dialog, "Processing Error", f"Error: {result['error_message']}")
                    
            except ImportError:
                self.loading_overlay.hide_loading()
                print("Remove Pulse module not found")
                QMessageBox.warning(dialog, "Module Not Found", "Remove Pulse module not found in dsp folder.")
            except Exception as e:
                self.loading_overlay.hide_loading()
                print(f"Error in Harmonic Extraction: {e}")
                QMessageBox.critical(dialog, "Error", f"Error in Harmonic Extraction:\n{str(e)}")

        def run_anomaly_detection(spectrogram_data):
            try:
                # Debug: Check if the module exists
                import sys
                import os
                print(f"Python path: {sys.path}")
                print(f"Current working directory: {os.getcwd()}")
                print(f"Looking for module at: {os.path.join(os.getcwd(), 'dsp', 'anomaly_detection.py')}")
                print(f"Module exists: {os.path.exists(os.path.join(os.getcwd(), 'dsp', 'anomaly_detection.py'))}")
                print(f"__init__.py exists: {os.path.exists(os.path.join(os.getcwd(), 'dsp', '__init__.py'))}")
                
                from dsp.anomaly_detection import process_anomaly_detection
                fp, img = spectrogram_data
                print(f"Running Anomaly Detection on: {os.path.basename(fp)}")
                
                # Show processing overlay
                self.loading_overlay.show_loading("Processing anomaly detection...")
                QApplication.processEvents()
                
                # Get current spectrogram from cache if available
                cached_data = self.audio_cache.get(fp)
                mel_spec = cached_data['spectrogram'] if cached_data else None
                
                # Process anomaly detection
                result = process_anomaly_detection(fp, mel_spec)
                
                if result['status'] == 'success':
                    # Update viewer with anomaly detection results
                    processed_data = cached_data.copy() if cached_data else {
                        'file_path': fp,
                        'audio_data': None,
                        'sample_rate': None,
                        'spectrogram': result['spectrogram'],
                        'duration_min': None
                    }
                    
                    processed_data.update({
                        'is_processed': True,
                        'processing_info': {
                            'method': 'Anomaly Detection',
                            'anomaly_results': result
                        }
                    })
                    
                    self.loading_overlay.update_progress(90, "Finalizing visualization...")
                    QApplication.processEvents()
                    
                    # Update the display with anomaly detection results
                    update_viewer_display(processed_data)
                    
                    self.loading_overlay.hide_loading()
                    
                    # Show results summary
                    selected_count = len(result['selected_indices'])
                    info_msg = (
                        f"Anomaly Detection completed!\n\n"
                        f"Found {selected_count} anomalous segments "
                        f"(top {result['top_percent']}%)\n\n"
                        f"The spectrogram now highlights detected anomalies."
                    )
                    QMessageBox.information(dialog, "Anomaly Detection Complete", info_msg)
                else:
                    self.loading_overlay.hide_loading()
                    QMessageBox.critical(dialog, "Processing Error", f"Error: {result['error_message']}")
                    
            except ImportError as ie:
                self.loading_overlay.hide_loading()
                print(f"Import error details: {ie}")
                print("Anomaly Detection module not found")
                QMessageBox.warning(dialog, "Module Not Found", f"Anomaly Detection module not found in dsp folder.\nError: {ie}")
            except Exception as e:
                self.loading_overlay.hide_loading()
                print(f"Error in Anomaly Detection: {e}")
                QMessageBox.critical(dialog, "Error", f"Error in Anomaly Detection:\n{str(e)}")

        remove_pulse_btn.clicked.connect(lambda: run_remove_pulse(spectrograms[index]))
        anomaly_detection_btn.clicked.connect(lambda: run_anomaly_detection(spectrograms[index]))

        # Add gain control
        gain_layout = QHBoxLayout()
        gain_label = QLabel("Spectrogram gain (dB):")
        gain_slider = QSlider(Qt.Horizontal)
        gain_slider.setRange(0, 40)
        gain_slider.setValue(0)
        gain_value = QLabel("0 dB")
        gain_layout.addWidget(gain_label)
        gain_layout.addWidget(gain_slider, 1)
        gain_layout.addWidget(gain_value)
        main_layout.addLayout(gain_layout)

        index = 0
        current_processor = None
        
        # Show dialog immediately
        dialog.show()
        QApplication.processEvents()
        
        # Start background preloading immediately
        self._preload_cache(spectrograms[:min(3, len(spectrograms))])
        
        def update_with_loading(idx):
            """Update viewer with loading bar for new files"""
            nonlocal index, current_processor
            
            # Cancel any existing processing
            if current_processor and current_processor.isRunning():
                current_processor.cancel()
                current_processor.terminate()
                current_processor.wait()
            
            index = idx
            fp, _ = spectrograms[idx]
            filename = os.path.basename(fp)
            
            # Update window title with current file info
            dialog.setWindowTitle(f"Waveform & Spectrogram Viewer - File {idx + 1} of {len(spectrograms)}: {filename}")
            
            # Check if data is already cached
            cached_data = self.audio_cache.get(fp)
            if cached_data:
                # Use cached data immediately
                try:
                    update_viewer_display(cached_data)
                    prev_btn.setEnabled(idx > 0)
                    next_btn.setEnabled(idx < len(spectrograms) - 1)
                    self._preload_adjacent_files(spectrograms, idx)
                    return
                except Exception as e:
                    print(f"Error using cached data: {e}")
            
            # Show loading overlay BEFORE disabling buttons
            self.loading_overlay.show_loading(f"Loading {filename}...")
            
            # Disable navigation during loading
            prev_btn.setEnabled(False)
            next_btn.setEnabled(False)
            
            # Create and start processor
            current_processor = AudioProcessorWithProgress(fp, self.audio_cache)
            
            def on_progress(value, message):
                if self.loading_overlay.isVisible():
                    self.loading_overlay.update_progress(value, message)
            
            def on_complete(result):
                try:
                    update_viewer_display(result)
                    self.loading_overlay.hide_loading()
                    
                    # Re-enable navigation
                    prev_btn.setEnabled(idx > 0)
                    next_btn.setEnabled(idx < len(spectrograms) - 1)
                    
                    # Preload adjacent files in background
                    self._preload_adjacent_files(spectrograms, idx)
                    
                except Exception as e:
                    print(f"Error updating display: {e}")
                    self.loading_overlay.hide_loading()
                    QMessageBox.critical(dialog, "Error updating display:\n{error_msg}")
                    # Re-enable buttons even on error
                    prev_btn.setEnabled(idx > 0)
                    next_btn.setEnabled(idx < len(spectrograms) - 1)
            
            def on_error(error_msg):
                self.loading_overlay.hide_loading()
                prev_btn.setEnabled(idx > 0)
                next_btn.setEnabled(idx < len(spectrograms) - 1)
                QMessageBox.critical(dialog, "Processing Error", f"Failed to process audio:\n{error_msg}")
            
            current_processor.progress_updated.connect(on_progress)
            current_processor.processing_complete.connect(on_complete)
            current_processor.processing_error.connect(on_error)
            current_processor.start()
        
        def update_viewer_display(result):
            """Update the matplotlib display with processed data"""
            try:
                self.figure.clear()
                
                fp = result['file_path']
                y = result['audio_data']
                sr = result['sample_rate']
                mel_spec_db = result['spectrogram']
                duration_min = result['duration_min']
                is_processed = result.get('is_processed', False)
                processing_info = result.get('processing_info', {})
                
                ax1 = self.figure.add_subplot(2, 1, 1)
                ax2 = self.figure.add_subplot(2, 1, 2, sharex=ax1)
                
                # Waveform plot (only if audio data available)
                if y is not None:
                    time_axis = np.linspace(0, duration_min, len(y))
                    color = 'blue' if is_processed else 'black'
                    ax1.plot(time_axis, y, color=color, linewidth=0.5)
                    
                    y_min, y_max = y.min(), y.max()
                    y_range = y_max - y_min
                    if y_range < 1e-6:
                        padding = 0.1
                    else:
                        padding = y_range * 0.1
                    ax1.set_ylim(y_min - padding, y_max + padding)
                else:
                    # Hide waveform plot if no audio data
                    ax1.text(0.5, 0.5, 'Waveform not available', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax1.transAxes, fontsize=12, color='gray')
                    ax1.set_ylim(0, 1)
                
                ax1.set_ylabel('Amplitude')
                ax1.grid(True, alpha=0.3)
                
                # Update title based on processing status
                title_base = f'{os.path.basename(fp)}'
                if is_processed:
                    method = processing_info.get('method', 'Processed')
                    title = f'[{method}] {title_base}'
                    ax1.set_title(f'Processed Waveform & Spectrogram: {title}')
                else:
                    ax1.set_title(f'Waveform & Spectrogram: {title_base}')

                # Spectrogram plot with gain
                db_gain = gain_slider.value()
                mel_spec_display = mel_spec_db + db_gain if db_gain > 0 else mel_spec_db
                
                hop_length = 512
                spec_time_frames = mel_spec_display.shape[1]
                if sr:
                    spec_duration_min = spec_time_frames * hop_length / sr / 60.0
                else:
                    spec_duration_min = duration_min or 1.0
                
                # Use different colormap for processed audio
                cmap = 'plasma'
                
                im = ax2.imshow(
                    mel_spec_display,
                    aspect='auto',
                    origin='lower',
                    extent=[0, spec_duration_min, 0, MAX_FREQ],
                    cmap=cmap
                )
                
                # Add anomaly detection overlays if available
                if is_processed and processing_info.get('method') == 'Anomaly Detection':
                    anomaly_results = processing_info.get('anomaly_results', {})
                    selected_indices = anomaly_results.get('selected_indices', [])
                    segment_starts = anomaly_results.get('segment_starts', [])
                    segment_width = anomaly_results.get('segment_width', 0)
                    n_mels = anomaly_results.get('n_mels', mel_spec_display.shape[0])
                    scores = anomaly_results.get('scores', [])
                    
                    # Convert segment positions to time
                    if selected_indices and segment_starts and sr:
                        for idx in selected_indices:
                            start_frame = segment_starts[idx]
                            start_time = start_frame * hop_length / sr / 60.0
                            width_time = segment_width * hop_length / sr / 60.0
                            score = scores[idx] if idx < len(scores) else 0
                            
                            # Add rectangle overlay with same style as template
                            rect = Rectangle((start_time, 0), width_time, MAX_FREQ,
                                           edgecolor='orange', facecolor='orange', 
                                           alpha=0.35, linewidth=0.5)  # Match template style
                            ax2.add_patch(rect)
                            
                            # Add segment number and score label with same style as template
                            ax2.text(start_time + width_time/20, MAX_FREQ * 0.95, 
                                #    f"[S{idx+1}]\n",  # Match template format
                                   f"", 
                                   color='orange', fontsize=10, ha='left', va='top',
                                   bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

                # Configure plot appearance
                font_size = 8
                ax1.set_ylabel('Amplitude', fontsize=font_size)
                ax1.tick_params(axis='both', which='major', labelsize=font_size, length=6)
                ax1.set_xticklabels([])
                
                ax2.set_xlabel('Time (min)', fontsize=font_size)
                ax2.set_ylabel('Frequency (kHz)', fontsize=font_size)
                ax2.set_ylim(0, MAX_FREQ)
                ax2.set_xlim(0, spec_duration_min)
                
                freqs = [0, 2000, 4000, 6000, 10000, 12000, 14000, 16000]
                ax2.set_yticks(freqs)
                ax2.set_yticklabels([_hz_to_k_label(f) for f in freqs], fontsize=font_size)
                
                # Set time ticks
                if spec_duration_min <= 5:
                    time_interval = 1.0
                elif spec_duration_min <= 15:
                    time_interval = 2.0
                elif spec_duration_min <= 30:
                    time_interval = 5.0
                else:
                    time_interval = 10.0
                
                time_ticks = np.arange(0, spec_duration_min + time_interval/2, time_interval)
                ax2.set_xticks(time_ticks)
                ax2.set_xticklabels([f"{int(t)}" for t in time_ticks], fontsize=font_size)
                ax2.tick_params(axis='both', which='major', labelsize=font_size, length=6)
                ax1.set_xticks(time_ticks)
                if y is not None:
                    ax1.set_xlim(0, duration_min)
                
                self.figure.tight_layout()
                self.canvas.draw()
                
            except Exception as e:
                print(f"Error in update_viewer_display: {e}")
                import traceback
                traceback.print_exc()
        
        # Connect navigation
        prev_btn.clicked.connect(lambda: update_with_loading(index-1))
        next_btn.clicked.connect(lambda: update_with_loading(index+1))
        
        # Optimized gain slider
        def update_gain():
            gain_value.setText(f"{gain_slider.value()} dB")
            if not self.loading_overlay.isVisible():
                fp, _ = spectrograms[index]
                cached_data = self.audio_cache.get(fp)
                if cached_data:
                    update_viewer_display(cached_data)
        
        gain_slider.valueChanged.connect(update_gain)
        
        # Load first file immediately after showing dialog
        QTimer.singleShot(50, lambda: update_with_loading(0))
        
        dialog.exec()
        
        # Cleanup when dialog closes
        if current_processor and current_processor.isRunning():
            current_processor.cancel()
            current_processor.terminate()
            current_processor.wait()

    def _preload_cache(self, file_list):
        """Preload audio files into cache in background"""
        def preload_worker():
            for fp, _ in file_list:
                if not self.audio_cache.get(fp):  # Only load if not already cached
                    try:
                        processor = AudioProcessorWithProgress(fp, self.audio_cache)
                        processor.run()  # Run synchronously in background thread
                    except Exception as e:
                        print(f"Error preloading {fp}: {e}")
        
        # Run preloading in background thread
        threading.Thread(target=preload_worker, daemon=True).start()
    
    def _preload_adjacent_files(self, spectrograms, current_idx):
        """Preload files adjacent to current index in background"""
        adjacent_files = []
        
        # Add previous file
        if current_idx > 0:
            adjacent_files.append(spectrograms[current_idx - 1])
        
        # Add next file
        if current_idx < len(spectrograms) - 1:
            adjacent_files.append(spectrograms[current_idx + 1])
        
        # Add next 2 files for better prefetching
        for i in range(current_idx + 2, min(current_idx + 4, len(spectrograms))):
            adjacent_files.append(spectrograms[i])
        
        if adjacent_files:
            self._preload_cache(adjacent_files)

    def closeEvent(self, event):
        global _ws
        # Cancel any running audio processing
        if self.audio_processor and self.audio_processor.isRunning():
            self.audio_processor.terminate()
            self.audio_processor.wait()
        
        # Clear audio cache
        if hasattr(self, 'audio_cache'):
            self.audio_cache.clear()
            
        if platform.system() == "Windows" and _ws:
            try: 
                _ws.win_sparkle_cleanup()
            except: 
                pass
        for p in self._session_temp_files:
            try: 
                os.remove(p)
            except: 
                pass
        # Delete update log file
        try:
            if os.path.exists(log_path):
                os.remove(log_path)
        except Exception:
            pass
        super().closeEvent(event)

    def _on_download_cancelled(self):
        """Handle download cancellation from the progress dialog"""
        # Only handle cancellation if download hasn't completed
        if not hasattr(self, '_download_completed') or not self._download_completed:
            self._download_cancelled = True
            if hasattr(self, '_download_progress') and self._download_progress:
                self._download_progress.close()
                self._download_progress = None

if __name__ == "__main__":
    # Create QApplication instance
    app = QApplication(sys.argv)
    
    # Create and show main window
    window = Main()
    window.show()
    
    # Start event loop
    sys.exit(app.exec())