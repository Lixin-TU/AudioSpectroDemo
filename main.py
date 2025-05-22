import sys
import time
import os
import platform
import pathlib

import ctypes
import shutil

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
import subprocess
import tempfile
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

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
        """Create an update script that replaces old versioned executable with new one"""
        current_exe = current_app_info['exe_path']
        app_dir = current_app_info['app_dir']
        
        # Use the desired final executable name”
        new_exe_name = final_exe_name
        final_new_exe_path = os.path.join(str(app_dir), new_exe_name)
        
        if platform.system() == "Windows":
            script_content = f'''
# AudioSpectroDemo Self‑Update Script
Write-Host "Starting update process..."

# Give the main application enough time to shut down
Start-Sleep -Seconds 3

$currentExe = "{current_exe}"
$tempNewExe = "{new_exe_path}"
$finalNewExe = "{final_new_exe_path}"
$backupExe = "$currentExe.backup"

Write-Host "Current executable: $currentExe"
Write-Host "Downloaded executable: $tempNewExe"
Write-Host "Final new executable: $finalNewExe"

# Verify new executable exists and is valid
if (-not (Test-Path $tempNewExe)) {{
    Write-Host "Error: Downloaded executable not found at $tempNewExe"
    exit 1
}}

$newSize = (Get-Item $tempNewExe).Length
if ($newSize -lt 1000000) {{  # Less than 1MB seems too small
    Write-Host "Error: Downloaded executable seems too small ($newSize bytes)"
    exit 1
}}

Write-Host "Backing up current executable..."
try {{
    Copy-Item $currentExe $backupExe -Force
    Write-Host "Backup created successfully"
}} catch {{
    Write-Host "Error creating backup: $($_.Exception.Message)"
    exit 1
}}

Write-Host "Moving new executable to application directory..."
try {{
    # Move the new executable to the app directory with its new name
    Move-Item $tempNewExe $finalNewExe -Force
    Write-Host "New executable moved successfully"
}} catch {{
    Write-Host "Error moving new executable: $($_.Exception.Message)"
    exit 1
}}

Write-Host "Removing old executable..."
try {{
    Remove-Item $currentExe -Force
    Write-Host "Old executable removed successfully"
}} catch {{
    Write-Host "Warning: Could not remove old executable: $($_.Exception.Message)"
    # This is not critical, continue with launch
}}

Write-Host "Launching updated application..."
try {{
    Start-Process -FilePath $finalNewExe
    Write-Host "Updated application launched successfully"
}} catch {{
    Write-Host "Error launching updated application: $($_.Exception.Message)"
    # Try to restore backup if launch fails
    if (Test-Path $backupExe) {{
        Write-Host "Restoring backup due to launch failure..."
        Copy-Item $backupExe $currentExe -Force
        Remove-Item $finalNewExe -Force -ErrorAction SilentlyContinue
        Start-Process -FilePath $currentExe
        Write-Host "Backup restored and launched"
    }}
}}

# Clean up
Write-Host "Cleaning up temporary files..."
Start-Sleep -Seconds 2
Remove-Item $backupExe -Force -ErrorAction SilentlyContinue

# Remove this script
Remove-Item $MyInvocation.MyCommand.Path -Force -ErrorAction SilentlyContinue
Write-Host "Update process completed"
'''
            script_file = tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.ps1', 
                delete=False,
                encoding='utf-8'
            )
            script_file.write(script_content)
            script_file.close()
            return script_file.name, 'powershell'
        else:
            script_content = f'''#!/bin/bash
echo "Starting update process..."

sleep 3

CURRENT_EXE="{current_exe}"
TEMP_NEW_EXE="{new_exe_path}"
FINAL_NEW_EXE="{final_new_exe_path}"
BACKUP_EXE="$CURRENT_EXE.backup"

echo "Current executable: $CURRENT_EXE"
echo "Downloaded executable: $TEMP_NEW_EXE"
echo "Final new executable: $FINAL_NEW_EXE"

# Verify new executable exists and is valid
if [ ! -f "$TEMP_NEW_EXE" ]; then
    echo "Error: Downloaded executable not found at $TEMP_NEW_EXE"
    exit 1
fi

NEW_SIZE=$(stat -f%z "$TEMP_NEW_EXE" 2>/dev/null || stat -c%s "$TEMP_NEW_EXE" 2>/dev/null)
if [ "$NEW_SIZE" -lt 1000000 ]; then  # Less than 1MB seems too small
    echo "Error: Downloaded executable seems too small ($NEW_SIZE bytes)"
    exit 1
fi

echo "Backing up current executable..."
if ! cp "$CURRENT_EXE" "$BACKUP_EXE"; then
    echo "Error creating backup"
    exit 1
fi
echo "Backup created successfully"

echo "Moving new executable to application directory..."
if ! mv "$TEMP_NEW_EXE" "$FINAL_NEW_EXE"; then
    echo "Error moving new executable"
    exit 1
fi

chmod +x "$FINAL_NEW_EXE"
echo "New executable moved successfully"

echo "Removing old executable..."
if ! rm -f "$CURRENT_EXE"; then
    echo "Warning: Could not remove old executable"
    # This is not critical, continue with launch
fi
echo "Old executable removed"

echo "Launching updated application..."
if ! "$FINAL_NEW_EXE" &; then
    echo "Error launching updated application"
    # Try to restore backup if launch fails
    if [ -f "$BACKUP_EXE" ]; then
        echo "Restoring backup due to launch failure..."
        mv "$BACKUP_EXE" "$CURRENT_EXE"
        rm -f "$FINAL_NEW_EXE"
        chmod +x "$CURRENT_EXE"
        "$CURRENT_EXE" &
        echo "Backup restored and launched"
    fi
else
    echo "Updated application launched successfully"
fi

# Clean up
echo "Cleaning up temporary files..."
sleep 2
rm -f "$BACKUP_EXE"

# Remove this script itself
sleep 1
rm -f "$0"
echo "Update process completed"
'''
            script_file = tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.sh', 
                delete=False
            )
            script_file.write(script_content)
            script_file.close()
            os.chmod(script_file.name, 0o755)
            return script_file.name, 'bash'

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
        new_exe_name = os.path.basename(url)
        
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
            return
        
        # Create path for new executable in temp directory
        # Keep the original filename from URL for the download
        new_exe_name = os.path.basename(url)
        if not new_exe_name:
            new_exe_name = f"AudioSpectroDemo-v{new_version}.exe"
        
        temp_new_exe = tempfile.NamedTemporaryFile(
            suffix=f"_{new_exe_name}", 
            delete=False
        )
        temp_new_exe.close()
        new_exe_temp_path = temp_new_exe.name

        progress = QProgressDialog("Downloading update...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(False)
        progress.setValue(0)
        progress.canceled.connect(lambda: self._cancel_download(new_exe_temp_path))
        QApplication.processEvents()

        self._download_cancelled = False

        def _hook(count, block_size, total_size):
            if self._download_cancelled:
                raise Exception("Download cancelled")
            if total_size > 0:
                pct = int(count * block_size * 100 / total_size)
                progress.setValue(min(pct, 100))
                # Update progress text with download info
                downloaded_mb = (count * block_size) / (1024*1024)
                total_mb = total_size / (1024*1024)
                progress.setLabelText(f"Downloading {new_exe_name}... {downloaded_mb:.1f}/{total_mb:.1f} MB")
                QApplication.processEvents()

        try:
            # Add headers to avoid potential blocking
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'AudioSpectroDemo/0.2.18')
            
            urllib.request.urlretrieve(url, new_exe_temp_path, reporthook=_hook)
            progress.close()
            
            if self._download_cancelled:
                return
            
            # Verify downloaded file
            if not os.path.exists(new_exe_temp_path):
                raise Exception("Downloaded file is missing")
                
            file_size = os.path.getsize(new_exe_temp_path)
            if file_size == 0:
                raise Exception("Downloaded file is empty")
            elif file_size < 1000000:  # Less than 1MB seems too small for this app
                raise Exception(f"Downloaded file seems too small ({file_size} bytes)")
            
            print(f"Downloaded new executable to: {new_exe_temp_path}")
            print(f"File size: {file_size} bytes")
            print(f"New executable name: {new_exe_name}")
            print(f"Current executable: {current_app_info['exe_path']}")
            
            # Create and launch update script
            self._launch_update_process(new_exe_temp_path, current_app_info)
            
        except Exception as e:
            progress.close()
            if not self._download_cancelled:
                error_msg = str(e)
                if "HTTP Error" in error_msg:
                    error_msg += "\n\nThis might be a temporary network issue. Please try again later."
                elif "SSL" in error_msg or "certificate" in error_msg.lower():
                    error_msg += "\n\nThis might be a certificate issue. Please check your internet connection."
                QMessageBox.critical(self, "Update Error", f"Failed to download update:\n{error_msg}")
            try:
                os.remove(new_exe_temp_path)
            except:
                pass

    def _cancel_download(self, file_path):
        """Handle download cancellation"""
        self._download_cancelled = True
        try:
            os.remove(file_path)
        except:
            pass

    def _launch_update_process(self, new_exe_temp_path, current_app_info):
        """Launch the update process using a script"""
        # Determine the target filename of the new executable (without the temporary prefix)
        dest_exe_name = os.path.basename(
            self.current_update_info.get("download_url", os.path.basename(new_exe_temp_path))
        )
        try:
            # Create update script
            script_path, script_type = self._create_update_script(
                new_exe_temp_path, current_app_info, dest_exe_name
            )
            final_path = current_app_info['app_dir'] / dest_exe_name
            
            print(f"Created update script: {script_path}")
            print(f"Temp new executable: {new_exe_temp_path}")
            print(f"Final new executable: {final_path}")
            print(f"Current executable: {current_app_info['exe_path']}")
            print(f"New executable name: {dest_exe_name}")
            
            # Show final confirmation
            QMessageBox.information(
                self, 
                "Update Starting", 
                f"The update will now begin.\n\n"
                f"• Current: {current_app_info['exe_path'].name}\n"
                f"• New: {dest_exe_name}\n\n"
                "The application will close and the new version will start automatically."
            )
            
            # Prepare for shutdown
            self._prepare_for_update()
            
            # Launch update script
            if script_type == 'powershell':
                # Use PowerShell on Windows
                subprocess.Popen([
                    'powershell', 
                    '-WindowStyle', 'Hidden',
                    '-ExecutionPolicy', 'Bypass',
                    '-File', script_path
                ], creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                # Use bash on Unix systems
                subprocess.Popen(['/bin/bash', script_path])
            
            # Close the application after a short delay
            QTimer.singleShot(500, lambda: QApplication.quit())
            
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
        """Export spectrograms to PNG files"""
        for wav_path, img in spectrograms:
            norm = img - img.min()
            if norm.max() > 0:
                norm = norm / norm.max()
            inds = (norm * 255).astype(np.uint8)
            rgb = LUT[inds]
            hop = 512
            export_dir = os.path.join(os.path.dirname(wav_path), "spectrograms")
            os.makedirs(export_dir, exist_ok=True)
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
            plt.close(fig)

    def show_spectrogram_viewer(self, spectrograms):
        """Show the spectrogram viewer dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Spectrogram Viewer")
        main_layout = QVBoxLayout(dialog)
        pw = PlotWidget(background="w")
        main_layout.addWidget(pw)

        btn_layout = QHBoxLayout()

        # Previous navigation button
        prev_btn = QPushButton("Previous")

        # Disabled, circular "Detect" button (under development)
        detect_btn = QPushButton("")
        detect_btn.setEnabled(False)
        detect_btn.setFixedSize(50, 50)  # circle diameter (larger)
        detect_btn.setToolTip("Detect Anomaly Events (under development)")
        detect_btn.setStyleSheet(
            "border-radius:25px;"
            "background-color:#CCCCCC;"  # neutral grey
            "border: 1px solid #999999;"
        )

        # Next navigation button
        next_btn = QPushButton("Next")

        # Add buttons to layout: Previous | Detect | Next
        btn_layout.addWidget(prev_btn)
        btn_layout.addWidget(detect_btn)
        btn_layout.addWidget(next_btn)

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
            nonlocal index
            index = idx
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
        # Cancel any running audio processing
        if self.audio_processor and self.audio_processor.isRunning():
            self.audio_processor.terminate()
            self.audio_processor.wait()
            
        if platform.system()=="Windows" and _ws:
            try: _ws.win_sparkle_cleanup()
            except: pass
        for p in self._session_temp_files:
            try: os.remove(p)
            except: pass
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec())