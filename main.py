import sys
import time
import os
import platform
import pathlib
import logging  # Add this import

import ctypes
import shutil

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
import xml.etree.ElementTree as ET
import urllib.request
from urllib.parse import urlsplit, unquote
import subprocess
import tempfile
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import re

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
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QThread, QRectF
import pyqtgraph as pg
from pyqtgraph import PlotWidget, ImageItem

# Add matplotlib imports for the new viewer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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

def parse_appcast_xml(url):
    """Parse the appcast XML to get update information"""
    try:
        # Add timeout to prevent hanging
        with urllib.request.urlopen(url, timeout=15) as response:
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
        logging.error(f"Error parsing appcast: {e}", exc_info=True)
        return None

# ---------------------------------------------------------------------------
# Helper: safely extract a usable filename from a download URL (strips
# query‑string parameters like “?raw=true” that break Windows paths)
def filename_from_url(url: str, default: str = "update.exe") -> str:

    try:
        path = urlsplit(url).path         
        name = os.path.basename(unquote(path))
        return name or default
    except Exception:
        return default
# ---------------------------------------------------------------------------


def check_for_updates_async():  
    """Check for updates in a separate thread"""
    try:
        current_version = "0.2.25"
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

# Convert dimensions: H=6cm, W=17cm at 300 DPI for export only
# 1 inch = 2.54 cm, so: cm * 300 DPI / 2.54 = pixels
EXPORT_W = int(17 * 300 / 2.54)  # 17cm = ~2008 pixels at 300 DPI
EXPORT_H = int(6 * 300 / 2.54)   # 6cm = ~709 pixels at 300 DPI

# Original dimensions for viewer dialog
VIEWER_W = 800
VIEWER_H = 400

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AudioSpectro")
        self._session_temp_files: list[str] = []
        self.resize(900, 600)
        self.audio_processor = None
        self._download_cancelled = False

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
            _ws.win_sparkle_set_app_details("UBCO-ISDPRL", "AudioSpectroDemo", "0.2.25")
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
        progress.canceled.connect(self._on_download_cancelled)

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
            # Add headers to avoid potential blocking
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'AudioSpectroDemo/0.2.25')

            # Actually perform the download
            urllib.request.urlretrieve(url, new_exe_temp_path, reporthook=_hook)
            
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
                if "HTTP Error" in error_msg:
                    error_msg += "\n\nThis might be a temporary network issue. Please try again later."
                elif "SSL" in error_msg or "certificate" in error_msg.lower():
                    error_msg += "\n\nThis might be a certificate issue. Please check your internet connection."
                QMessageBox.critical(self, "Update Error", f"Failed to download update:\n{error_msg}")
            try:
                os.remove(new_exe_temp_path)
            except Exception as e:
                pass

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
        """Show the waveform and spectrogram viewer dialog using matplotlib"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Waveform & Spectrogram Viewer")
        main_layout = QVBoxLayout(dialog)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

        btn_layout = QHBoxLayout()

        # Previous navigation button
        prev_btn = QPushButton("Previous")
        prev_btn.setEnabled(False)

        # Analysis button with floating menu
        analysis_btn = QPushButton("Analysis")
        analysis_btn.setFixedSize(60, 60)  # Make it square for circular shape
        analysis_btn.setStyleSheet(
            "QPushButton {"
            "    background-color: #FF7777;"  # Lighter red
            "    color: white;"
            "    border: 2px solid white;"    # White border
            "    border-radius: 30px;"        # Circular shape (half of 60px)
            "    font-weight: bold;"
            "    font-size: 9px;"
            "}"
            "QPushButton:hover {"
            "    background-color: #FF9999;"  # Even lighter on hover
            "}"
            "QPushButton:pressed {"
            "    background-color: #FF5555;"  # Slightly darker when pressed
            "}"
        )

        # Next navigation button
        next_btn = QPushButton("Next")
        next_btn.setEnabled(len(spectrograms) > 1)

        btn_layout.addWidget(prev_btn)
        btn_layout.addWidget(analysis_btn)
        btn_layout.addWidget(next_btn)

        main_layout.addLayout(btn_layout)

        # Create fan-shaped analysis options that stay in the viewer
        analysis_options_layout = QHBoxLayout()
        
        # Create sub-option buttons
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
        anomaly_detection_btn.setEnabled(False)  # Disabled temporarily
        anomaly_detection_btn.setStyleSheet(
            "QPushButton {"
            "    background-color: #CCCCCC;"
            "    color: #666666;"
            "    border: 1px solid #999999;"
            "    border-radius: 15px;"
            "    font-size: 8px;"
            "}"
        )
        
        ai_anomaly_btn = QPushButton("AI Anomaly Detection")
        ai_anomaly_btn.setFixedSize(140, 30)
        ai_anomaly_btn.setEnabled(False)  # Disabled temporarily
        ai_anomaly_btn.setStyleSheet(
            "QPushButton {"
            "    background-color: #CCCCCC;"
            "    color: #666666;"
            "    border: 1px solid #999999;"
            "    border-radius: 15px;"
            "    font-size: 8px;"
            "}"
        )
        
        # Initially hide the analysis options
        remove_pulse_btn.setVisible(False)
        anomaly_detection_btn.setVisible(False)
        ai_anomaly_btn.setVisible(False)
        
        # Add buttons to layout with spacing for fan effect
        analysis_options_layout.addStretch(1)
        analysis_options_layout.addWidget(remove_pulse_btn)
        analysis_options_layout.addWidget(anomaly_detection_btn)
        analysis_options_layout.addWidget(ai_anomaly_btn)
        analysis_options_layout.addStretch(1)
        
        main_layout.addLayout(analysis_options_layout)
        
        # Track visibility state
        options_visible = False

        # Toggle analysis options visibility with fan animation effect
        def toggle_analysis_options():
            nonlocal options_visible
            options_visible = not options_visible
            
            if options_visible:
                # Show buttons with a fan-like reveal
                remove_pulse_btn.setVisible(True)
                anomaly_detection_btn.setVisible(True)
                ai_anomaly_btn.setVisible(True)
                analysis_btn.setText("Close")
            else:
                # Hide buttons
                remove_pulse_btn.setVisible(False)
                anomaly_detection_btn.setVisible(False)
                ai_anomaly_btn.setVisible(False)
                analysis_btn.setText("Analysis")

        # Connect analysis button to toggle options
        analysis_btn.clicked.connect(toggle_analysis_options)

        # Analysis function implementations
        def run_remove_pulse(spectrogram_data):
            try:
                from dsp.remove_pulse import process_remove_pulse
                fp, img = spectrogram_data
                print(f"Running Remove Pulse on: {os.path.basename(fp)}")
                result = process_remove_pulse(fp)
                print(f"Remove Pulse completed: {result}")
                # Show result in a message box
                QMessageBox.information(dialog, "Remove Pulse", f"Processing completed!\n\nResults:\n{result}")
            except ImportError:
                print("Remove Pulse module not found")
                QMessageBox.warning(dialog, "Module Not Found", "Remove Pulse module not found in dsp folder.")
            except Exception as e:
                print(f"Error in Remove Pulse: {e}")
                QMessageBox.critical(dialog, "Error", f"Error in Remove Pulse:\n{str(e)}")

        def run_anomaly_detection(spectrogram_data):
            # Disabled for now
            QMessageBox.information(dialog, "Feature Disabled", "Anomaly Detection is temporarily disabled.")

        def run_ai_anomaly_detection(spectrogram_data):
            # Disabled for now
            QMessageBox.information(dialog, "Feature Disabled", "AI Anomaly Detection is temporarily disabled.")

        # Connect only the enabled button
        remove_pulse_btn.clicked.connect(lambda: run_remove_pulse(spectrograms[index]))

        # Add gain control for spectrogram
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
        
        def update(idx):
            nonlocal index
            index = idx
            
            try:
                # Clear the figure
                self.figure.clear()
                
                fp, _ = spectrograms[idx] # Ignore pre-computed spectrogram, generate fresh
                dialog.setWindowTitle(os.path.basename(fp))
                
                # Load audio for waveform
                y, sr = librosa.load(fp, sr=TARGET_SR, mono=True)
                
                # Calculate timing
                duration_min = len(y) / sr / 60.0
                
                print(f"File: {os.path.basename(fp)}")
                print(f"Audio: {len(y)} samples, {y.min():.3f} to {y.max():.3f}")
                print(f"Duration: {duration_min:.2f} min")
                
                # Create subplots - waveform on top, spectrogram on bottom
                ax1 = self.figure.add_subplot(2, 1, 1)
                ax2 = self.figure.add_subplot(2, 1, 2, sharex=ax1)  # Share x-axis
                
                # --- WAVEFORM PLOT ---
                time_axis = np.linspace(0, duration_min, len(y))
                ax1.plot(time_axis, y, color='black', linewidth=0.5)
                ax1.set_ylabel('Amplitude')
                ax1.grid(True, alpha=0.3)
                ax1.set_xlim(0, duration_min)
                
                # Set proper amplitude range with padding
                y_min, y_max = y.min(), y.max()
                y_range = y_max - y_min
                if y_range < 1e-6:
                    padding = 0.1
                else:
                    padding = y_range * 0.1
                ax1.set_ylim(y_min - padding, y_max + padding)
                ax1.set_title(f'Waveform & Spectrogram: {os.path.basename(fp)}')
                
                # --- SPECTROGRAM PLOT ---
                # Generate mel spectrogram using librosa
                n_mels = 256
                hop_length = 512
                n_fft = 2048
                
                # Compute mel spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=y, 
                    sr=sr, 
                    n_mels=n_mels,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    fmin=MIN_FREQ,
                    fmax=MAX_FREQ
                )
                
                # Convert to dB scale
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Apply gain from slider
                db_gain = gain_slider.value()
                if db_gain > 0:
                    mel_spec_db = mel_spec_db + db_gain
                
                # Create time axis for spectrogram (in minutes)
                spec_time_frames = mel_spec_db.shape[1]
                spec_duration_min = spec_time_frames * hop_length / sr / 60.0
                
                # Display spectrogram
                im = ax2.imshow(
                    mel_spec_db,
                    aspect='auto',
                    origin='lower',  # Low frequencies at bottom
                    extent=[0, spec_duration_min, 0, MAX_FREQ],
                    cmap='plasma'
                )
                
                # Set consistent font size for all text elements
                font_size = 8
                
                # Configure waveform plot
                ax1.set_ylabel('Amplitude', fontsize=font_size)
                ax1.set_title(f'Waveform & Spectrogram: {os.path.basename(fp)}', fontsize=font_size)
                ax1.tick_params(axis='both', which='major', labelsize=font_size, length=6)
                ax1.set_xticklabels([])  # Remove labels from top plot
                
                # Configure spectrogram plot
                ax2.set_xlabel('Time (min)', fontsize=font_size)
                ax2.set_ylabel('Frequency (kHz)', fontsize=font_size)
                ax2.set_ylim(0, MAX_FREQ)
                ax2.set_xlim(0, spec_duration_min)
                
                # Set frequency ticks to match export format
                freqs = [0, 2000, 4000, 6000, 10000, 12000, 14000, 16000]
                ax2.set_yticks(freqs)
                ax2.set_yticklabels([_hz_to_k_label(f) for f in freqs], fontsize=font_size)
                
                # Set time ticks with consistent intervals
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
                
                # Apply same time ticks to waveform
                ax1.set_xticks(time_ticks)

                # Adjust layout and refresh
                self.figure.tight_layout()
                self.canvas.draw()
        
                # Force canvas update
                self.canvas.flush_events()
                
                # Enable/disable navigation buttons
                prev_btn.setEnabled(idx > 0)
                next_btn.setEnabled(idx < len(spectrograms) - 1)
                
                print(f"Matplotlib waveform and spectrogram created successfully")
                print(f"Spectrogram shape: {mel_spec_db.shape}")
                print(f"Time alignment: waveform={duration_min:.2f}min, spec={spec_duration_min:.2f}min")
                
            except Exception as e:
                print(f"Error updating viewer: {e}")
                import traceback
                traceback.print_exc()

        # Connect UI elements
        prev_btn.clicked.connect(lambda: update(index-1))
        next_btn.clicked.connect(lambda: update(index+1))
        gain_slider.valueChanged.connect(lambda v: (gain_value.setText(f"{v} dB"), update(index)))
        
        # Set initial view
        update(0)
        
        # Set dialog size and show
        dialog.resize(1000, 600)
        dialog.exec()

    def closeEvent(self, event):
        global _ws
        # Cancel any running audio processing
        if self.audio_processor and self.audio_processor.isRunning():
            self.audio_processor.terminate()
            self.audio_processor.wait()
            
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
            # If we have a temp file path, try to clean it up later
            # Not cleaning up immediately as the file might still be in use
            pass
        else:
            pass

if __name__ == "__main__":
    try:
        # Set up basic error handling
        app = QApplication(sys.argv)
        window = Main()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        # If the app fails to start, show an error dialog
        error_msg = f"Failed to start application: {str(e)}\n\nPlease check the update_log.txt file for details."
        print(error_msg)
        logging.error(f"Application failed to start: {e}", exc_info=True)
        
        try:
            if QApplication.instance():
                QMessageBox.critical(None, "Application Error", error_msg)
        except:
            pass
        sys.exit(1)