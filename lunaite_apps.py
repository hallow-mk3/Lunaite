"""
Lunaite AI 10T — Windows Desktop Application & System Automation Suite
======================================================================
Provides autonomous control for Windows laptop applications:
  1. Spotify       - Search & direct auto-play, media control, playlist launcher
  2. Discord       - Launch, quick switcher navigation to DMs/users/channels
  3. Outlook       - Read unread emails, check calendar, compose/send drafts (via COM / mailto / web)
  4. File Explorer - Open special folders (Downloads, Docs, etc.), search files, reveal files
  5. System Apps   - Launch Chrome, Edge, VS Code, Notepad, Terminal, Settings, Calc, Steam, Games
  6. Media Control - Volume up/down/mute, play/pause global media keys, lock screen

Created by Swasthik Shetty for Lunaite AI.
"""

import os
import sys
import time
import subprocess
import urllib.parse
import shutil
import glob
from typing import Dict, Any, List, Optional

# Try importing win32com for Outlook and SAPI
try:
    import win32com.client
    HAS_WIN32COM = True
except ImportError:
    HAS_WIN32COM = False

# Try importing pyautogui for media keys / window actions
try:
    import pyautogui
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False


# ─── 1. SPOTIFY INTEGRATION (SEARCH & AUTO-PLAY) ──────────────────────────────

def spotify_open() -> str:
    """Launch or focus Spotify."""
    try:
        os.system('start "" "spotify:"')
        return "Spotify launched / focused."
    except Exception as e:
        return f"[Spotify Error]: {e}"


def spotify_search_and_play(query: str) -> str:
    """Search and immediately play the top track on Spotify."""
    try:
        q = query.strip()
        encoded = urllib.parse.quote(q)
        uri = f"spotify:search:{encoded}"
        
        # 1. Open search in Spotify desktop app
        subprocess.run(["cmd", "/c", "start", "", uri], shell=False, check=False)
        
        # 2. Automate selecting and playing the first track result
        if HAS_PYAUTOGUI:
            time.sleep(1.2)
            # Press Enter to play top result if focused, or Tab + Enter
            pyautogui.press("enter")
            time.sleep(0.3)
            pyautogui.press("playpause")
            
        return f"Playing top search result for '{q}' in Spotify."
    except Exception as e:
        return f"[Spotify Play Error]: {e}"


def spotify_media_action(action: str) -> str:
    """Perform media action: playpause, next, prev, volumeup, volumedown, mute."""
    action = action.lower().strip()
    try:
        if HAS_PYAUTOGUI:
            if action in ["play", "pause", "playpause", "toggle"]:
                pyautogui.press("playpause")
                return "Toggled Spotify / Media Play/Pause."
            elif action in ["next", "skip", "nexttrack"]:
                pyautogui.press("nexttrack")
                return "Skipped to next track."
            elif action in ["prev", "previous", "prevtrack"]:
                pyautogui.press("prevtrack")
                return "Returned to previous track."
            elif action in ["volup", "volumeup", "louder"]:
                for _ in range(5):
                    pyautogui.press("volumeup")
                return "Increased system volume."
            elif action in ["voldown", "volumedown", "quieter"]:
                for _ in range(5):
                    pyautogui.press("volumedown")
                return "Decreased system volume."
            elif action in ["mute", "unmute"]:
                pyautogui.press("volumemute")
                return "Toggled audio mute."

        # Fallback via PowerShell
        ps_keys = {
            "playpause": "0xB3",
            "next": "0xB0",
            "prev": "0xB1",
            "volumeup": "0xAF",
            "volumedown": "0xAE",
            "mute": "0xAD"
        }
        key_code = ps_keys.get(action, "0xB3")
        ps_cmd = (
            f"$wscript = New-Object -ComObject WScript.Shell; "
            f"$wscript.SendKeys([char]{key_code})"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True)
        return f"Dispatched media action: {action}"
    except Exception as e:
        return f"[Media Action Error]: {e}"


# ─── 2. DISCORD INTEGRATION (USER/DM QUICK SWITCHER) ──────────────────────────

def discord_open(channel_or_user: Optional[str] = None) -> str:
    """Launch Discord and navigate to a user DM, channel, or guild."""
    try:
        # Launch/focus Discord app first
        discord_local = os.path.expandvars(r"%LOCALAPPDATA%\Discord\Update.exe --processStart Discord.exe")
        if os.path.exists(os.path.expandvars(r"%LOCALAPPDATA%\Discord")):
            subprocess.Popen(f'"{discord_local}"', shell=True)
        else:
            subprocess.run(["cmd", "/c", "start", "", "discord:"], shell=False, check=False)

        if channel_or_user and channel_or_user.strip():
            target = channel_or_user.strip()
            
            # If it's a numeric ID or discord:// path
            if target.startswith("discord://") or target.isdigit():
                uri = target if target.startswith("discord://") else f"discord://-/users/{target}"
                subprocess.run(["cmd", "/c", "start", "", uri], shell=False, check=False)
                return f"Navigated Discord to: {target}"
            
            # Use Discord's Quick Switcher (Ctrl+K) to find and open the user's DM
            if HAS_PYAUTOGUI:
                time.sleep(1.0)
                pyautogui.hotkey("ctrl", "k")
                time.sleep(0.3)
                pyautogui.write(target, interval=0.03)
                time.sleep(0.4)
                pyautogui.press("enter")
                return f"Opened Discord Quick Switcher & navigated to messages with '{target}'."
                
            return f"Focused Discord for user '{target}'."
            
        return "Discord launched & brought to focus."
    except Exception as e:
        return f"[Discord Error]: {e}"


# ─── 3. OUTLOOK INTEGRATION ───────────────────────────────────────────────────

def outlook_open() -> str:
    """Launch Microsoft Outlook desktop client or Outlook Web."""
    try:
        subprocess.run(["cmd", "/c", "start", "", "outlook"], shell=False, check=False)
        return "Outlook launched."
    except Exception:
        subprocess.run(["cmd", "/c", "start", "", "https://outlook.live.com"], shell=False, check=False)
        return "Opened Outlook in browser."


def outlook_get_unread_emails(max_count: int = 5) -> str:
    """Read unread emails from Outlook, with automatic web fallback."""
    if HAS_WIN32COM:
        try:
            outlook = win32com.client.Dispatch("Outlook.Application")
            namespace = outlook.GetNamespace("MAPI")
            inbox = namespace.GetDefaultFolder(6)  # 6 = olFolderInbox
            messages = inbox.Items
            messages.Sort("[ReceivedTime]", True)
            
            unread_items = []
            count = 0
            for msg in messages:
                try:
                    if msg.UnRead:
                        sender = getattr(msg, "SenderName", "Unknown")
                        subject = getattr(msg, "Subject", "(No Subject)")
                        received = getattr(msg, "ReceivedTime", "")
                        body_preview = str(getattr(msg, "Body", ""))[:120].replace("\r", " ").replace("\n", " ").strip()
                        
                        unread_items.append(
                            f"📧 [Unread {count+1}]\n"
                            f"  From: {sender}\n"
                            f"  Subject: {subject}\n"
                            f"  Date: {received}\n"
                            f"  Preview: {body_preview}..."
                        )
                        count += 1
                        if count >= max_count:
                            break
                except Exception:
                    continue

            if unread_items:
                return f"Found {len(unread_items)} unread email(s) in Outlook:\n\n" + "\n\n".join(unread_items)
            else:
                return "No unread emails in your Outlook inbox."
        except Exception:
            pass

    # Clean fallback: open Outlook web or desktop
    subprocess.run(["cmd", "/c", "start", "", "https://outlook.live.com/mail/inbox"], shell=False, check=False)
    return "Outlook desktop client is not actively signed in. Opened your Outlook Web Inbox in the browser to view unread messages."


def outlook_compose_email(to: str, subject: str, body: str, send: bool = False) -> str:
    """Create a draft or send an email in Outlook."""
    if HAS_WIN32COM:
        try:
            outlook = win32com.client.Dispatch("Outlook.Application")
            mail = outlook.CreateItem(0)  # 0 = olMailItem
            mail.To = to
            mail.Subject = subject
            mail.Body = body
            if send:
                mail.Send()
                return f"Successfully sent email to {to} with subject '{subject}'."
            else:
                mail.Save()
                mail.Display(True)
                return f"Created Outlook draft to {to} with subject '{subject}'."
        except Exception:
            pass

    # Fallback via mailto URI
    try:
        mailto_url = f"mailto:{urllib.parse.quote(to)}?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"
        os.system(f'start "" "{mailto_url}"')
        return f"Opened mail composer to send message to {to}."
    except Exception as e:
        return f"[Mail Compose Error]: {e}"


def outlook_get_calendar_events(days_ahead: int = 3) -> str:
    """Get upcoming calendar events from Outlook."""
    if HAS_WIN32COM:
        try:
            outlook = win32com.client.Dispatch("Outlook.Application")
            namespace = outlook.GetNamespace("MAPI")
            calendar = namespace.GetDefaultFolder(9)  # 9 = olFolderCalendar
            items = calendar.Items
            items.Sort("[Start]")
            items.IncludeRecurrences = True
            
            events = []
            import datetime
            now = datetime.datetime.now()
            end_time = now + datetime.timedelta(days=days_ahead)
            
            date_filter = f"[Start] >= '{now.strftime('%m/%d/%Y %H:%M')}' AND [Start] <= '{end_time.strftime('%m/%d/%Y %H:%M')}'"
            filtered_items = items.Restrict(date_filter)
            
            for item in filtered_items:
                try:
                    subj = getattr(item, "Subject", "Event")
                    start = getattr(item, "Start", "")
                    loc = getattr(item, "Location", "")
                    events.append(f"📅 {subj} | Start: {start} | Location: {loc if loc else 'N/A'}")
                    if len(events) >= 10:
                        break
                except Exception:
                    continue
                    
            if events:
                return f"Upcoming Calendar Events (Next {days_ahead} days):\n\n" + "\n".join(events)
            return f"No events scheduled on your Outlook calendar for the next {days_ahead} days."
        except Exception:
            pass
            
    subprocess.run(["cmd", "/c", "start", "", "https://outlook.live.com/calendar"], shell=False, check=False)
    return "Opened your Outlook Calendar in the browser."


# ─── 4. FILE EXPLORER INTEGRATION ─────────────────────────────────────────────

SPECIAL_FOLDERS = {
    "downloads": os.path.expandvars(r"%USERPROFILE%\Downloads"),
    "documents": os.path.expandvars(r"%USERPROFILE%\Documents"),
    "docs": os.path.expandvars(r"%USERPROFILE%\Documents"),
    "desktop": os.path.expandvars(r"%USERPROFILE%\Desktop"),
    "pictures": os.path.expandvars(r"%USERPROFILE%\Pictures"),
    "music": os.path.expandvars(r"%USERPROFILE%\Music"),
    "videos": os.path.expandvars(r"%USERPROFILE%\Videos"),
    "c drive": "C:\\",
    "c:": "C:\\",
    "root": "C:\\",
    "project": os.path.abspath(".")
}

def explorer_open(folder_or_path: str = "downloads") -> str:
    """Open a folder in Windows File Explorer."""
    path_key = folder_or_path.lower().strip()
    target = SPECIAL_FOLDERS.get(path_key, folder_or_path.strip())
    target = os.path.expandvars(os.path.expanduser(target))
    
    try:
        if os.path.exists(target):
            subprocess.run(["explorer", target], check=False)
            return f"Opened folder in File Explorer: {target}"
        else:
            subprocess.run(["explorer"], check=False)
            return f"Opened default File Explorer (folder '{target}' not found)."
    except Exception as e:
        return f"[Explorer Error]: {e}"


def explorer_search_files(query: str, folder_key: str = "downloads") -> str:
    """Search for files in a folder by name substring."""
    folder_path = SPECIAL_FOLDERS.get(folder_key.lower().strip(), folder_key.strip())
    folder_path = os.path.expandvars(os.path.expanduser(folder_path))
    
    if not os.path.exists(folder_path):
        return f"[Search Error]: Folder does not exist: {folder_path}"
        
    try:
        matches = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if query.lower() in file.lower():
                    full_p = os.path.join(root, file)
                    size_kb = round(os.path.getsize(full_p) / 1024, 1)
                    matches.append(f"📄 {file} ({size_kb} KB)\n   Path: {full_p}")
                    if len(matches) >= 15:
                        break
            if len(matches) >= 15:
                break
                
        if matches:
            return f"Found {len(matches)} matching file(s) for '{query}' in {folder_path}:\n\n" + "\n\n".join(matches)
        return f"No files matching '{query}' found in {folder_path}."
    except Exception as e:
        return f"[File Search Error]: {e}"


# ─── 5. GENERAL APPLICATION LAUNCHER & CONTROLLER ─────────────────────────────

APP_REGISTRY = {
    "spotify": {"cmd": "spotify:", "name": "Spotify"},
    "discord": {"cmd": "discord:", "name": "Discord"},
    "outlook": {"cmd": "outlook", "name": "Microsoft Outlook"},
    "explorer": {"cmd": "explorer", "name": "File Explorer"},
    "files": {"cmd": "explorer", "name": "File Explorer"},
    "chrome": {"cmd": "chrome", "name": "Google Chrome"},
    "edge": {"cmd": "msedge", "name": "Microsoft Edge"},
    "browser": {"cmd": "https://google.com", "name": "Web Browser"},
    "vscode": {"cmd": "code", "name": "Visual Studio Code"},
    "vs code": {"cmd": "code", "name": "Visual Studio Code"},
    "code": {"cmd": "code", "name": "Visual Studio Code"},
    "notepad": {"cmd": "notepad", "name": "Notepad"},
    "calculator": {"cmd": "calc", "name": "Calculator"},
    "calc": {"cmd": "calc", "name": "Calculator"},
    "terminal": {"cmd": "wt", "name": "Windows Terminal"},
    "cmd": {"cmd": "cmd", "name": "Command Prompt"},
    "powershell": {"cmd": "powershell", "name": "PowerShell"},
    "settings": {"cmd": "ms-settings:", "name": "Windows Settings"},
    "task manager": {"cmd": "taskmgr", "name": "Task Manager"},
    "taskmgr": {"cmd": "taskmgr", "name": "Task Manager"},
    "steam": {"cmd": "steam:", "name": "Steam"},
    "whatsapp": {"cmd": "whatsapp:", "name": "WhatsApp"},
    "black myth": {"cmd": "steam://rungameid/2358720", "name": "Black Myth: Wukong (Steam)"},
    "wukong": {"cmd": "steam://rungameid/2358720", "name": "Black Myth: Wukong (Steam)"}
}

def open_application(app_name: str) -> str:
    """Launch or focus any desktop application cleanly."""
    app_key = app_name.lower().strip()
    
    for key, info in APP_REGISTRY.items():
        if key == app_key or key in app_key or app_key in key:
            cmd = info["cmd"]
            try:
                subprocess.Popen(f'start "" "{cmd}"', shell=True)
                return f"Successfully opened {info['name']}."
            except Exception as e:
                return f"Failed to open {info['name']}: {e}"

    try:
        subprocess.Popen(f'start "" "{app_name}"', shell=True)
        return f"Launched application: {app_name}"
    except Exception as e:
        return f"[App Launch Error for '{app_name}']: {e}"


def close_application(app_name: str) -> str:
    """Close an application process."""
    proc_name = app_name.strip()
    if not proc_name.endswith(".exe"):
        proc_name += ".exe"
    try:
        res = subprocess.run(["taskkill", "/F", "/IM", proc_name], capture_output=True, text=True)
        if res.returncode == 0:
            return f"Successfully closed {proc_name}."
        return f"Could not close {proc_name}: {res.stderr.strip() or res.stdout.strip()}"
    except Exception as e:
        return f"[Close App Error]: {e}"


def list_running_applications() -> str:
    """List main active window titles and top processes."""
    try:
        ps_cmd = 'Get-Process | Where-Object {$_.MainWindowTitle -ne ""} | Select-Object ProcessName, MainWindowTitle | Format-Table -AutoSize'
        result = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True, timeout=8)
        out = result.stdout.strip()
        return f"**Active Running Applications:**\n\n```\n{out}\n```" if out else "No visible window processes found."
    except Exception as e:
        return f"[Process List Error]: {e}"


# ─── 6. SYSTEM CONTROLS ───────────────────────────────────────────────────────

def system_control(action: str) -> str:
    """Perform system-level control actions."""
    act = action.lower().strip()
    try:
        if act == "lock":
            subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"])
            return "Windows workstation locked."
        elif act in ["mute", "unmute"]:
            return spotify_media_action("mute")
        elif act in ["volume_up", "volup"]:
            return spotify_media_action("volumeup")
        elif act in ["volume_down", "voldown"]:
            return spotify_media_action("volumedown")
        elif act == "playpause":
            return spotify_media_action("playpause")
        elif act == "settings":
            subprocess.run(["cmd", "/c", "start", "", "ms-settings:"], shell=False)
            return "Opened Windows Settings."
        elif act == "screenshot":
            result = take_screenshot()
            if result.get("error"):
                return f"[Screenshot Error]: {result['error']}"
            return f"📸 Screenshot saved to: {result['path']}"
        return f"Unknown system control action: {action}"
    except Exception as e:
        return f"[System Control Error]: {e}"


# ─── 7. DEEP SYSTEM ACCESS ────────────────────────────────────────────────────

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import base64
    HAS_BASE64 = True
except ImportError:
    HAS_BASE64 = False


def get_system_telemetry() -> dict:
    """Return live system telemetry: CPU, RAM, disk, battery, uptime."""
    data = {
        "cpu_percent": 0.0,
        "ram_percent": 0.0,
        "ram_used_gb": 0.0,
        "ram_total_gb": 0.0,
        "disk_percent": 0.0,
        "disk_used_gb": 0.0,
        "disk_total_gb": 0.0,
        "battery_percent": None,
        "battery_charging": None,
        "uptime_seconds": 0,
        "process_count": 0,
        "cpu_cores": 0,
        "cpu_freq_mhz": 0.0,
    }
    if not HAS_PSUTIL:
        return data
    try:
        data["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        data["cpu_cores"] = psutil.cpu_count(logical=True)
        freq = psutil.cpu_freq()
        if freq:
            data["cpu_freq_mhz"] = round(freq.current, 1)

        mem = psutil.virtual_memory()
        data["ram_percent"] = mem.percent
        data["ram_used_gb"] = round(mem.used / (1024 ** 3), 2)
        data["ram_total_gb"] = round(mem.total / (1024 ** 3), 2)

        disk = psutil.disk_usage("C:\\")
        data["disk_percent"] = disk.percent
        data["disk_used_gb"] = round(disk.used / (1024 ** 3), 1)
        data["disk_total_gb"] = round(disk.total / (1024 ** 3), 1)

        battery = psutil.sensors_battery()
        if battery:
            data["battery_percent"] = round(battery.percent, 1)
            data["battery_charging"] = battery.power_plugged

        data["uptime_seconds"] = int(time.time() - psutil.boot_time())
        data["process_count"] = len(psutil.pids())
    except Exception:
        pass
    return data


def take_screenshot(save_dir: str = ".") -> dict:
    """Capture a screenshot and return path + base64 thumbnail."""
    if not HAS_PYAUTOGUI:
        return {"error": "pyautogui not installed", "path": None, "thumb_b64": None}
    try:
        ts = int(time.time())
        save_path = os.path.abspath(os.path.join(save_dir, f"lunaite_screenshot_{ts}.png"))
        img = pyautogui.screenshot()
        img.save(save_path)

        thumb = img.copy()
        thumb.thumbnail((320, 180))
        import io
        buf = io.BytesIO()
        thumb.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {"path": save_path, "thumb_b64": b64, "width": img.width, "height": img.height}
    except Exception as e:
        return {"error": str(e), "path": None, "thumb_b64": None}


def read_clipboard() -> str:
    """Read text content from the Windows clipboard."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-Clipboard"],
            capture_output=True, text=True, timeout=5
        )
        content = result.stdout.strip()
        if not content:
            return "[Clipboard is empty or contains non-text content]"
        return f"**Clipboard Contents:**\n{content}"
    except Exception as e:
        return f"[Clipboard Read Error]: {e}"


def write_clipboard(text: str) -> str:
    """Write text to the Windows clipboard."""
    try:
        escaped = text.replace("'", "''")
        ps_cmd = f"Set-Clipboard -Value '{escaped}'"
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=5
        )
        return f"Copied {len(text)} characters to clipboard."
    except Exception as e:
        return f"[Clipboard Write Error]: {e}"


def run_powershell(command: str, timeout: int = 20) -> str:
    """Execute an arbitrary PowerShell command and return its output."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
            capture_output=True, text=True, timeout=timeout
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if err and not out:
            return f"[PS Error]:\n{err}"
        elif err and out:
            return f"[PS Output]:\n{out}\n\n[PS Warnings]:\n{err}"
        return out if out else "[Command completed with no output]"
    except subprocess.TimeoutExpired:
        return f"[PS Timeout]: Command exceeded {timeout}s"
    except Exception as e:
        return f"[PS Execution Error]: {e}"


def read_file_content(path: str, max_lines: int = 100) -> str:
    """Read and return the content of a local file."""
    try:
        expanded = os.path.expandvars(os.path.expanduser(path.strip()))
        if not os.path.exists(expanded):
            return f"[File Not Found]: {expanded}"
        if not os.path.isfile(expanded):
            return f"[Not a File]: {expanded}"
        size = os.path.getsize(expanded)
        if size > 500_000:
            return f"[File Too Large]: {expanded} ({round(size/1024,1)} KB). Max 500KB."
        with open(expanded, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if len(lines) > max_lines:
            content = "".join(lines[:max_lines])
            return f"**File: {expanded}** ({len(lines)} lines, showing first {max_lines}):\n\n{content}\n...[truncated]"
        return f"**File: {expanded}**\n\n{''.join(lines)}"
    except Exception as e:
        return f"[File Read Error]: {e}"


def write_file_content(path: str, content: str) -> str:
    """Write content to a local file (creates parent dirs if needed)."""
    try:
        expanded = os.path.expandvars(os.path.expanduser(path.strip()))
        parent = os.path.dirname(expanded)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(expanded, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to: {expanded}"
    except Exception as e:
        return f"[File Write Error]: {e}"


def kill_process(name: str) -> str:
    """Kill a running process by name or PID."""
    name = name.strip()
    try:
        if name.isdigit():
            cmd = ["taskkill", "/F", "/PID", name]
        else:
            if not name.lower().endswith(".exe"):
                name += ".exe"
            cmd = ["taskkill", "/F", "/IM", name]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            return f"Terminated process: {name}"
        return f"[Kill Failed]: {res.stderr.strip() or res.stdout.strip()}"
    except Exception as e:
        return f"[Kill Process Error]: {e}"


def open_url_in_browser(url: str) -> str:
    """Open any URL in the user's default browser."""
    url = url.strip()
    try:
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url
        subprocess.run(["cmd", "/c", "start", "", url], shell=False, check=False)
        return f"Opened in browser: {url}"
    except Exception as e:
        return f"[Browser Open Error]: {e}"


def list_directory(path: str = "downloads", max_items: int = 30) -> str:
    """List directory contents with sizes and modification times."""
    import datetime as _dt
    folder = SPECIAL_FOLDERS.get(path.lower().strip(), path.strip())
    folder = os.path.expandvars(os.path.expanduser(folder))
    if not os.path.isdir(folder):
        return f"[Not a Directory]: {folder}"
    try:
        items = []
        for entry in sorted(os.scandir(folder), key=lambda e: e.stat().st_mtime, reverse=True)[:max_items]:
            stat = entry.stat()
            mtime = _dt.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            if entry.is_dir():
                items.append(f"📁 {entry.name}/  [{mtime}]")
            else:
                size_kb = round(stat.st_size / 1024, 1)
                items.append(f"📄 {entry.name}  ({size_kb} KB)  [{mtime}]")
        if not items:
            return f"Directory is empty: {folder}"
        return f"**Contents of {folder}** ({len(items)} items):\n\n" + "\n".join(items)
    except Exception as e:
        return f"[Directory List Error]: {e}"


# ─── 8. UNIFIED INTENT ROUTER ─────────────────────────────────────────────────

def execute_app_intent(intent: str, **kwargs) -> str:
    """Execute unified application intent."""
    intent = intent.lower().strip()

    if "spotify" in intent:
        action = kwargs.get("action", "open")
        query = kwargs.get("query", "")
        if "play" in intent or action == "play":
            if query:
                return spotify_search_and_play(query)
            return spotify_media_action("playpause")
        elif "next" in intent or action == "next":
            return spotify_media_action("next")
        elif "prev" in intent or action == "prev":
            return spotify_media_action("prev")
        elif "pause" in intent or action == "pause":
            return spotify_media_action("playpause")
        return spotify_open()

    elif "discord" in intent:
        return discord_open(kwargs.get("target", None))

    elif "outlook" in intent or "email" in intent or "mail" in intent:
        action = kwargs.get("action", "unread")
        if action == "unread" or "unread" in intent or "check" in intent or "read" in intent:
            return outlook_get_unread_emails()
        elif action == "calendar" or "calendar" in intent or "schedule" in intent:
            return outlook_get_calendar_events()
        elif action in ("send", "compose") or "send" in intent or "draft" in intent:
            return outlook_compose_email(
                kwargs.get("to", ""), kwargs.get("subject", "Message from Lunaite"),
                kwargs.get("body", ""), send=kwargs.get("send", False)
            )
        return outlook_open()

    elif "explorer" in intent or "folder" in intent:
        target = kwargs.get("target", kwargs.get("path", "downloads"))
        query = kwargs.get("query", "")
        return explorer_search_files(query, target) if query else explorer_open(target)

    elif "screenshot" in intent:
        result = take_screenshot()
        if result.get("error"):
            return f"[Screenshot Error]: {result['error']}"
        return f"📸 Screenshot captured: {result['path']}"

    elif "clipboard" in intent:
        action = kwargs.get("action", "read")
        if action == "write":
            return write_clipboard(kwargs.get("text", ""))
        return read_clipboard()

    elif "powershell" in intent or "run command" in intent:
        return run_powershell(kwargs.get("command", "Get-Date"))

    elif "read file" in intent:
        return read_file_content(kwargs.get("path", ""))

    elif "write file" in intent:
        return write_file_content(kwargs.get("path", ""), kwargs.get("content", ""))

    elif "kill" in intent or "terminate" in intent:
        return kill_process(kwargs.get("process", kwargs.get("name", "")))

    elif "browser" in intent or "open url" in intent or "open website" in intent:
        return open_url_in_browser(kwargs.get("url", "https://google.com"))

    elif "list" in intent and ("file" in intent or "dir" in intent or "folder" in intent):
        return list_directory(kwargs.get("path", "downloads"))

    elif "app" in intent or "open" in intent or "launch" in intent:
        app_name = kwargs.get("app_name", kwargs.get("target",
            intent.replace("open", "").replace("launch", "").strip()))
        return open_application(app_name)

    elif "system" in intent or "media" in intent:
        return system_control(kwargs.get("action", "playpause"))

    return "No matching application intent detected."
