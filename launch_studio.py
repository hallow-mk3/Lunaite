"""
Lunaite AI 10B — Interactive Studio Launcher
=============================================
Launches the 1-Click Lunaite 10B Training Studio and opens the web application.

Usage:
    python launch_studio.py
"""

import os
import sys
import time
import webbrowser
import threading
import uvicorn

def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://localhost:8000")

def main():
    print("=" * 65)
    print("  🌙 LUNAITE AI 10B — Interactive Dataset & Training Studio")
    print("  Created by Swasthik Shetty")
    print("=" * 65)
    print("\n  [+] Starting Web Studio Server on http://localhost:8000 ...")
    print("  [+] Opening Web Studio in your default browser...")
    print("  [+] Press Ctrl+C in this terminal anytime to stop the studio.\n")

    threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run("lunaite_studio_server:app", host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()
