def download_update(self):
    """Download and install update - replaces old version with new versioned executable"""
    info = getattr(self, 'current_update_info', {})
    url = info.get('download_url')
    if not url:
        QMessageBox.warning(self, "Update Error", "No download URL available")
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
    if not new_exe_name:
        new_exe_name = f"AudioSpectroDemo-v{new_version}.exe"

    # Use a safe suffix (replace chars that Windows forbids, e.g. '?')
    safe_suffix = "_" + new_exe_name.replace("?", "_").replace(":", "_").replace("|", "_")
    temp_new_exe = tempfile.NamedTemporaryFile(
        suffix=safe_suffix,
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
        
        print(f"Starting download from: {url}")
        print(f"Downloading to: {new_exe_temp_path}")
        
        urllib.request.urlretrieve(url, new_exe_temp_path, reporthook=_hook)
        progress.close()
        
        if self._download_cancelled:
            print("Download was cancelled")
            return
        
        # Verify downloaded file
        if not os.path.exists(new_exe_temp_path):
            raise Exception("Downloaded file is missing")
            
        file_size = os.path.getsize(new_exe_temp_path)
        print(f"Downloaded file size: {file_size} bytes")
        
        if file_size == 0:
            raise Exception("Downloaded file is empty")
        elif file_size < 1000000:  # Less than 1MB seems too small for this app
            raise Exception(f"Downloaded file seems too small ({file_size} bytes)")

        # If running from source (not frozen), we cannot auto‑replace the script.
        if not current_app_info['is_frozen']:
            QMessageBox.information(
                self,
                "Update downloaded",
                "The update was downloaded to:\n"
                f"{new_exe_temp_path}\n\n"
                "Because you're running the Python source, "
                "the auto‑update step is skipped.\n"
                "Run the packaged EXE to enable one‑click updates."
            )
            return

        print(f"Downloaded new executable to: {new_exe_temp_path}")
        print(f"New executable name: {new_exe_name}")
        print(f"Current executable: {current_app_info['exe_path']}")

        # Show confirmation before attempting update
        reply = QMessageBox.question(
            self,
            "Install Update",
            f"Download completed successfully!\n\n"
            f"File size: {file_size:,} bytes\n"
            f"Ready to install update and restart the application.\n\n"
            f"Continue with installation?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply != QMessageBox.Yes:
            try:
                os.remove(new_exe_temp_path)
                print("Update cancelled, temp file removed")
            except Exception as e:
                print(f"Failed to remove temp file: {e}")
            return

        # Attempt the update process
        final_path = current_app_info['app_dir'] / new_exe_name
        print(f"Target path: {final_path}")
        
        try:
            # Check if target directory is writable
            test_file = current_app_info['app_dir'] / "test_write.tmp"
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                print("Directory is writable")
            except Exception as e:
                raise Exception(f"Cannot write to application directory: {e}")
            
            # Check if we can move/copy the file
            print("Attempting to move new executable into place...")
            
            # If target already exists, try to remove it first
            if final_path.exists():
                print(f"Target file already exists: {final_path}")
                try:
                    os.remove(final_path)
                    print("Removed existing target file")
                except Exception as e:
                    print(f"Warning: Could not remove existing file: {e}")
            
            # Move the new file into place
            try:
                shutil.move(new_exe_temp_path, final_path)
                print("Successfully moved new executable")
            except Exception as move_error:
                print(f"Move failed: {move_error}, trying copy instead...")
                try:
                    shutil.copy2(new_exe_temp_path, final_path)
                    os.remove(new_exe_temp_path)
                    print("Successfully copied new executable")
                except Exception as copy_error:
                    raise Exception(f"Both move and copy failed. Move: {move_error}, Copy: {copy_error}")

            # Verify the new file was created successfully
            if not final_path.exists():
                raise Exception("New executable was not created at target location")
                
            new_file_size = os.path.getsize(final_path)
            if new_file_size != file_size:
                raise Exception(f"File size mismatch: expected {file_size}, got {new_file_size}")
                
            print(f"New executable verified at: {final_path}")

            # Show success message
            QMessageBox.information(
                self,
                "Update Installed",
                f"Update installed successfully!\n\n"
                f"New version: {new_exe_name}\n"
                f"Location: {final_path}\n\n"
                f"The application will now restart with the new version."
            )

            # Prepare for shutdown and launch new version
            print("Preparing to launch new version and exit...")
            self._prepare_for_update()
            
            # Launch the new version
            try:
                subprocess.Popen([str(final_path)], cwd=str(current_app_info['app_dir']))
                print(f"Launched new version: {final_path}")
            except Exception as e:
                print(f"Failed to launch new version: {e}")
                QMessageBox.critical(
                    self, 
                    "Launch Error", 
                    f"Update was installed but failed to launch new version:\n{e}\n\n"
                    f"Please manually run: {final_path}"
                )
                return

            # Clean up old versions and exit
            print("Cleaning up and exiting...")
            QTimer.singleShot(1000, lambda: (
                self._cleanup_old_versions(),
                os._exit(0)
            ))

        except Exception as e:
            error_msg = f"Failed to install update: {e}"
            print(error_msg)
            QMessageBox.critical(self, "Installation Error", error_msg)
            try:
                os.remove(new_exe_temp_path)
                print("Cleaned up temp file after installation failure")
            except Exception as cleanup_error:
                print(f"Failed to clean up temp file: {cleanup_error}")

    except Exception as e:
        progress.close()
        if not self._download_cancelled:
            error_msg = str(e)
            print(f"Download error: {error_msg}")
            
            if "HTTP Error" in error_msg:
                error_msg += "\n\nThis might be a temporary network issue. Please try again later."
            elif "SSL" in error_msg or "certificate" in error_msg.lower():
                error_msg += "\n\nThis might be a certificate issue. Please check your internet connection."
            elif "URLError" in error_msg:
                error_msg += "\n\nPlease check your internet connection and try again."
                
            QMessageBox.critical(self, "Download Error", f"Failed to download update:\n{error_msg}")
        else:
            print("Download was cancelled by user")
            
        try:
            os.remove(new_exe_temp_path)
            print("Cleaned up temp file after download failure")
        except Exception as cleanup_error:
            print(f"Failed to clean up temp file: {cleanup_error}")