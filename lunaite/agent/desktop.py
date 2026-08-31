"""
Lunaite Architecture — System & Desktop Automation Suite
========================================================
Provides autonomous system telemetry and desktop control capabilities:
- Hardware Telemetry (CPU, Memory, GPU VRAM, Disk)
- Screen Capture & Clipboard I/O
- PowerShell & Command Execution
- Local File Inspection & Editing
- Process & Application Automation

Author: Swasthik Shetty <swasthik.mk3@gmail.com>
License: MIT
"""

import os
import sys
import time
import shutil
import subprocess
import urllib.parse
from typing import Dict, Any, List, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pyautogui
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False


def _get_hidden_kwargs() -> Dict[str, Any]:
    """Return subprocess flags to prevent console windows from flashing on Windows."""
    kwargs: Dict[str, Any] = {}
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0  # SW_HIDE
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kwargs


def get_system_telemetry() -> Dict[str, Any]:
    """Retrieve real-time hardware telemetry."""
    data = {
        "cpu_percent": 0.0,
        "ram_percent": 0.0,
        "ram_used_gb": 0.0,
        "ram_total_gb": 0.0,
        "gpu_name": "N/A",
        "gpu_vram_used_gb": 0.0,
        "gpu_vram_total_gb": 0.0,
        "disk_free_gb": 0.0,
        "platform": sys.platform
    }
    if HAS_PSUTIL:
        try:
            data["cpu_percent"] = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            data["ram_percent"] = mem.percent
            data["ram_used_gb"] = round(mem.used / (1024**3), 2)
            data["ram_total_gb"] = round(mem.total / (1024**3), 2)

            disk = psutil.disk_usage(os.getcwd() if sys.platform != "win32" else "C:\\")
            data["disk_free_gb"] = round(disk.free / (1024**3), 2)
        except Exception:
            pass

    # Check GPU via torch if available
    try:
        import torch
        if torch.cuda.is_available():
            data["gpu_name"] = torch.cuda.get_device_name(0)
            data["gpu_vram_used_gb"] = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
            data["gpu_vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    except Exception:
        pass

    return data


def take_screenshot(save_dir: str = "./data") -> Dict[str, Any]:
    """Capture current display screen and save to disk."""
    if not HAS_PYAUTOGUI:
        return {"error": "pyautogui package not installed"}
    try:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)
        img = pyautogui.screenshot()
        img.save(filepath)
        return {"success": True, "path": os.path.abspath(filepath)}
    except Exception as e:
        return {"error": str(e)}


def read_clipboard() -> str:
    """Read text currently copied to system clipboard."""
    if HAS_PYAUTOGUI:
        try:
            import pyperclip
            return pyperclip.paste()
        except Exception:
            pass
    if sys.platform == "win32":
        try:
            res = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", "Get-Clipboard"],
                capture_output=True,
                text=True,
                timeout=3,
                **_get_hidden_kwargs()
            )
            return res.stdout.strip()
        except Exception as e:
            return f"[Clipboard Read Error]: {e}"
    return "Clipboard reading not supported on this platform."


def write_clipboard(text: str) -> str:
    """Copy text to system clipboard."""
    if sys.platform == "win32":
        try:
            p = subprocess.Popen(
                ["clip"],
                stdin=subprocess.PIPE,
                shell=False,
                **_get_hidden_kwargs()
            )
            p.communicate(text.encode("utf-8"))
            return "Copied to clipboard."
        except Exception as e:
            return f"[Clipboard Write Error]: {e}"
    return "Clipboard writing not supported on this platform."


def run_powershell(cmd: str, timeout: int = 15) -> str:
    """Execute PowerShell command safely without flashing console windows."""
    try:
        if sys.platform == "win32":
            res = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                capture_output=True,
                text=True,
                timeout=timeout,
                **_get_hidden_kwargs()
            )
            out = res.stdout.strip() or res.stderr.strip()
            return out if out else "[Command completed with zero output]"
        else:
            res = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            out = res.stdout.strip() or res.stderr.strip()
            return out if out else "[Command completed with zero output]"
    except Exception as e:
        return f"[Command Error]: {e}"


def read_file_content(filepath: str, max_bytes: int = 20000) -> str:
    """Read text file safely."""
    try:
        if not os.path.exists(filepath):
            return f"Error: File '{filepath}' does not exist."
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_bytes)
        return content
    except Exception as e:
        return f"[File Read Error]: {e}"


def write_file_content(filepath: str, content: str) -> str:
    """Write text content to file."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to '{filepath}'."
    except Exception as e:
        return f"[File Write Error]: {e}"


def list_running_applications() -> List[str]:
    """List names of active user processes."""
    if not HAS_PSUTIL:
        return ["psutil not installed"]
    apps = set()
    for p in psutil.process_iter(['name']):
        try:
            name = p.info.get('name')
            if name and not name.lower().startswith(('svchost', 'system', 'registry', 'smss', 'csrss', 'wininit')):
                apps.add(name)
        except Exception:
            pass
    return sorted(list(apps))[:30]


def kill_process(target: str) -> str:
    """Terminate process by name or PID without flashing console windows."""
    if sys.platform == "win32":
        try:
            if target.isdigit():
                subprocess.run(
                    ["taskkill", "/F", "/PID", target],
                    check=False,
                    **_get_hidden_kwargs()
                )
            else:
                name = target if target.endswith(".exe") else f"{target}.exe"
                subprocess.run(
                    ["taskkill", "/F", "/IM", name],
                    check=False,
                    **_get_hidden_kwargs()
                )
            return f"Terminated process: {target}"
        except Exception as e:
            return f"[Kill Error]: {e}"
    return "Kill not supported."
