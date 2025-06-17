import os
import sys
import platform
import threading
import logging
import urllib.request
from urllib.parse import urlsplit, unquote
import xml.etree.ElementTree as ET
from PySide6.QtCore import QObject, Signal

class UpdateChecker(QObject):
    update_available = Signal(dict)
    update_not_available = Signal(str)
    update_error = Signal(str)

update_checker = UpdateChecker()

def parse_appcast_xml(url: str) -> dict:
    """Parse the appcast XML to get update information"""
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            xml_data = resp.read()
        root = ET.fromstring(xml_data)
        items = root.findall('.//item')
        if not items:
            return None
        li = items[0]
        encl = li.find('enclosure')
        if encl is not None:
            length_attr = encl.get('length')
            length = length_attr or "0"
            download_url = encl.get('url') or ""
        else:
            length = "0"
            download_url = ""
        return {
            'version': (li.findtext('title') or "Unknown").strip(),
            'description': li.findtext('description') or "",
            'pub_date': li.findtext('pubDate') or "",
            'download_url': download_url,
            'file_size': int(length) if length.isdigit() else 0
        }
    except Exception as e:
        logging.error(f"Error parsing appcast: {e}", exc_info=True)
        return None

def filename_from_url(url: str, default: str = "update.exe") -> str:
    """Safely extract a usable filename from a download URL"""
    try:
        path = urlsplit(url).path
        name = os.path.basename(unquote(path))
        return name or default
    except Exception:
        return default

def check_for_updates_async():
    """Check for updates in a separate thread"""
    try:
        current = "0.2.26"
        appcast = "https://raw.githubusercontent.com/Lixin-TU/AudioSpectroDemo/main/appcast.xml"
        info = parse_appcast_xml(appcast)
        if not info:
            update_checker.update_error.emit("Failed to fetch update information")
            return
        latest = info['version'].replace('Version ', '').strip()
        if latest != current:
            update_checker.update_available.emit(info)
        else:
            update_checker.update_not_available.emit(current)
    except Exception as e:
        update_checker.update_error.emit(f"Update check failed: {e}")
