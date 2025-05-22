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

logging.basicConfig(
    filename=log_path,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Replace print statements with logging
def log_print(message):
    print(message)  # Still print to console when available
    logging.info(message)  # Also log to file

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
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QThread
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
        current_version = "0.2.18"
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
            _ws.win_sparkle_set_app_details("UBCO-ISDPRL", "AudioSpectroDemo", "0.2.18")
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
        log_print("\n--- STARTING UPDATE PROCESS ---")
        info = getattr(self, 'current_update_info', {})
        url = info.get('download_url')
        if not url:
            log_print("Error: No download URL found in update info")
            return

        # Extract version info for display
        new_version = info.get('version', 'Unknown').replace('Version ', '').strip()
        file_size_mb = info.get('file_size', 0) / (1024*1024)
        size_text = f" ({file_size_mb:.1f} MB)" if file_size_mb > 0 else ""
        log_print(f"Update info: version={new_version}, size={file_size_mb:.1f}MB")

        # Get current app info to show current version
        current_app_info = self._get_current_app_info()
        current_name = current_app_info['exe_path'].name
        new_exe_name = filename_from_url(url)
        log_print(f"Current app: {current_name} (frozen: {current_app_info['is_frozen']})")
        log_print(f"New exe name will be: {new_exe_name}")

        reply = QMessageBox.question(
            self,
            "Update Confirmation",
            f"This will update from {current_name} to {new_exe_name}{size_text}.\n\n"
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
            log_print("Update cancelled by user")
            return

        # Create path for new executable in temp directory
        if not new_exe_name:
            new_exe_name = f"AudioSpectroDemo-v{new_version}.exe"
            log_print(f"Generated exe name: {new_exe_name}")

        # Use a safe suffix
        safe_suffix = "_" + new_exe_name.replace("?", "_")
        temp_new_exe = tempfile.NamedTemporaryFile(
            suffix=safe_suffix,
            delete=False
        )
        temp_new_exe.close()
        new_exe_temp_path = temp_new_exe.name
        log_print(f"Temporary download location: {new_exe_temp_path}")

        # IMPORTANT: Reset the download cancelled flag before starting
        self._download_cancelled = False
        
        # Create a progress dialog with a more controlled cancel behavior
        progress = QProgressDialog("Initializing download...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.setValue(0)
        
        # Store progress dialog reference to access it later
        self._download_progress = progress
        self._download_completed = False  # New flag to track download completion
        
        # Only connect once to avoid duplicate signals
        progress.canceled.connect(self._on_download_cancelled)

        # Reset any existing download path
        self._current_download_path = new_exe_temp_path

        def _safely_close_progress():
            if hasattr(self, '_download_progress') and self._download_progress:
                try:
                    # Block signals before closing to prevent unwanted cancel signals
                    if self._download_progress is not None:
                        self._download_progress.blockSignals(True)
                        self._download_progress.close()
                        self._download_progress = None
                        log_print("Progress dialog closed safely")
                except Exception as e:
                    log_print(f"Error closing progress: {e}")
        
        # Implement a direct download with better timeout handling
        try:
            log_print(f"Starting download from URL: {url}")
            progress.setLabelText("Connecting to server...")
            QApplication.processEvents()
            
            # Use a session with timeout
            import socket
            import urllib3
            
            # Set default socket timeout
            default_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(30)  # 30 second timeout
            
            http = urllib3.PoolManager(
                timeout=urllib3.Timeout(connect=30.0, read=60.0),
                retries=urllib3.Retry(3)
            )
            
            # Try to get content length first with a HEAD request
            try:
                head_response = http.request('HEAD', url)
                content_length = int(head_response.headers.get('Content-Length', 0))
                log_print(f"Content length from HEAD: {content_length} bytes")
            except Exception as e:
                log_print(f"Error getting content length: {e}")
                content_length = 0
            
            # Now start the actual download
            progress.setLabelText("Starting download...")
            QApplication.processEvents()
            
            response = http.request(
                'GET',
                url,
                preload_content=False,  # Stream the response
                headers={
                    'User-Agent': 'AudioSpectroDemo/0.2.18'
                }
            )
            
            if response.status != 200:
                raise Exception(f"HTTP error: {response.status} {response.reason}")
            
            # Get content length from the response if we couldn't get it before
            if content_length == 0:
                content_length = int(response.headers.get('Content-Length', 0))
                log_print(f"Content length from GET: {content_length} bytes")
            
            # Initialize a counter for downloaded bytes
            downloaded = 0
            
            # Open the output file
            with open(new_exe_temp_path, 'wb') as out_file:
                # Read the response in chunks and write to the output file
                chunk_size = 8192
                last_update_time = time.time()
                
                progress.setLabelText(f"Downloading {new_exe_name}...")
                QApplication.processEvents()
                
                for chunk in response.stream(chunk_size):
                    if self._download_cancelled:
                        response.release_conn()
                        raise Exception("Download cancelled by user")
                    
                    if chunk:
                        out_file.write(chunk)
                        downloaded += len(chunk)
                        
                        # Update progress every 0.5 seconds
                        now = time.time()
                        if now - last_update_time > 0.5:
                            last_update_time = now
                            if content_length > 0:
                                percent = min(int((downloaded / content_length) * 100), 100)
                                progress.setValue(percent)
                                
                                # Add download speed information
                                downloaded_mb = downloaded / (1024*1024)
                                total_mb = content_length / (1024*1024)
                                progress.setLabelText(f"Downloading {new_exe_name}... {downloaded_mb:.1f}/{total_mb:.1f} MB")
                            else:
                                # If we don't know the total size, show bytes downloaded
                                downloaded_mb = downloaded / (1024*1024)
                                progress.setLabelText(f"Downloading {new_exe_name}... {downloaded_mb:.1f} MB")
                            
                            QApplication.processEvents()
            
            # Clean up the response
            response.release_conn()
            
            # Reset socket timeout to default
            socket.setdefaulttimeout(default_timeout)
            
            # Mark download as completed
            self._download_completed = True
            log_print("Download completed successfully")
            
            # Now safely close the progress dialog
            _safely_close_progress()
            
            # Force UI update
            QApplication.processEvents()
            log_print("UI updated after download")
            
            # Additional safeguard: verify the download worked and file exists
            if not os.path.exists(new_exe_temp_path):
                log_print("ERROR: Download file is missing")
                QMessageBox.critical(self, "Update Error", "Downloaded file is missing.")
                return
                
            # Verify downloaded file
            file_size = os.path.getsize(new_exe_temp_path)
            log_print(f"Verifying download: file_size={file_size} bytes")
            if file_size == 0:
                log_print("Downloaded file is empty")
                QMessageBox.critical(self, "Update Error", "Downloaded file is empty.")
                return
            elif file_size < 1000000:  # Less than 1MB seems too small for this app
                log_print(f"Downloaded file seems too small ({file_size} bytes)")
                QMessageBox.critical(self, "Update Error", f"Downloaded file seems too small ({file_size} bytes).")
                return

            # Debug output
            log_print(f"Download details:")
            log_print(f"  File: {new_exe_temp_path}")
            log_print(f"  Size: {file_size} bytes")
            log_print(f"  Target name: {new_exe_name}")
            
            # Continue with the existing code to handle the downloaded file
            # ...existing code...
            
        except Exception as e:
            # Ensure progress dialog is closed on error
            _safely_close_progress()
            
            # Force UI update
            QApplication.processEvents()
            
            if not self._download_cancelled:
                error_msg = str(e)
                log_print(f"Download error: {error_msg}")
                
                # Provide more specific error messages based on the exception
                if "HTTP error" in error_msg:
                    error_msg += "\n\nThis might be a temporary server issue. Please try again later."
                elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                    error_msg = f"Connection timed out: {error_msg}\n\nPlease check your internet connection and try again."
                elif "SSL" in error_msg or "certificate" in error_msg.lower():
                    error_msg += "\n\nThis might be a certificate issue. Please check your internet security settings."
                elif "Name or service not known" in error_msg or "getaddrinfo failed" in error_msg:
                    error_msg = "Cannot reach update server. Please check your internet connection."
                
                QMessageBox.critical(self, "Update Error", f"Failed to download update:\n{error_msg}")
            else:
                log_print("Download was cancelled by user")
            
            # Clean up temp file
            try:
                if os.path.exists(new_exe_temp_path):
                    os.remove(new_exe_temp_path)
                    log_print(f"Removed temp file: {new_exe_temp_path}")
            except Exception as e:
                log_print(f"Error removing temp file: {e}")
            return

    def _on_download_cancelled(self):
        """Handle download cancellation from the progress dialog"""
        # Only handle cancellation if download hasn't completed
        if not hasattr(self, '_download_completed') or not self._download_completed:
            log_print("Download cancel requested by user from progress dialog")
            self._download_cancelled = True
            
            # If we have a temp file path, try to clean it up later
            # Not cleaning up immediately as the file might still be in use
            if hasattr(self, '_current_download_path') and self._current_download_path:
                log_print(f"Will clean up temp file later: {self._current_download_path}")
        else:
            log_print("Ignoring cancellation signal - download already completed")