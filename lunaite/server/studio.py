"""
Lunaite Architecture — Interactive Studio Backend Server
=========================================================
FastAPI application powering the Lunaite Studio Web UI.
Supports dynamic model selection, dataset inspection, parameter calculation,
asynchronous training with WebSocket loss telemetry, and interactive streaming chat.

Author: Swasthik Shetty <swasthik.mk3@gmail.com>
License: MIT
"""

import os
import sys
import json
import time
import asyncio
import subprocess
import threading
from typing import Dict, Any, List, Optional

try:
    from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from ..config import LunaiteConfig
from ..core.architecture import calculate_architecture_parameters
from ..agent.desktop import get_system_telemetry, take_screenshot, read_clipboard, write_clipboard, run_powershell
from ..agent.tools import web_search, fetch_url, wiki_lookup, fetch_weather
from ..train.dataset import load_dataset_file, generate_preset_dataset
from ..models.wrapper import wrap

if HAS_FASTAPI:
    app = FastAPI(
        title="Lunaite Architecture Studio",
        description="Unified Web Interface and API for Lunaite Neural Architecture",
        version="3.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    DATA_DIR = os.path.abspath("./data")
    os.makedirs(DATA_DIR, exist_ok=True)
    WEB_DIR = os.path.abspath("./web")
    os.makedirs(WEB_DIR, exist_ok=True)

    # ─── Training State Manager ───────────────────────────────────────────────
    class StudioTrainingState:
        def __init__(self):
            self.is_training = False
            self.progress = 0.0
            self.epoch = 0
            self.total_epochs = 3
            self.step = 0
            self.total_steps = 0
            self.loss = 0.0
            self.learning_rate = 2e-4
            self.eta_seconds = 0
            self.status_message = "Idle"
            self.logs = []
            self.history = []
            self.active_sockets = []

        def register_ws(self, ws: WebSocket):
            self.active_sockets.append(ws)

        def unregister_ws(self, ws: WebSocket):
            if ws in self.active_sockets:
                self.active_sockets.remove(ws)

        async def broadcast(self, payload: Dict[str, Any]):
            for ws in list(self.active_sockets):
                try:
                    await ws.send_json(payload)
                except Exception:
                    self.unregister_ws(ws)

    studio_state = StudioTrainingState()

    class ChatRequest(BaseModel):
        prompt: str
        model: Optional[str] = "lunaite-ai"
        deliberate: Optional[bool] = False
        enable_agent: Optional[bool] = True

    class ArchitectureCalcRequest(BaseModel):
        base_parameters: int = 7_000_000_000
        d_model: int = 4096
        num_layers: int = 32
        num_experts: int = 8
        expert_dim: int = 1024
        lora_rank: int = 64

    # ─── API Endpoints ────────────────────────────────────────────────────────

    @app.get("/api/models")
    async def get_available_models():
        """List local Ollama models and standard foundation models."""
        models = ["lunaite-ai", "qwen2.5:7b", "llama3.1:8b", "mistral:7b", "gemma2:9b", "phi3:mini"]
        try:
            res = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=3)
            lines = res.stdout.strip().split("\n")
            if len(lines) > 1:
                ollama_models = [line.split()[0] for line in lines[1:] if line.strip()]
                if ollama_models:
                    models = ollama_models
        except Exception:
            pass
        return {"models": models, "default": models[0] if models else "lunaite-ai"}

    @app.get("/api/telemetry")
    async def telemetry_endpoint():
        return get_system_telemetry()

    @app.post("/api/architecture/calculate")
    async def calculate_arch(req: ArchitectureCalcRequest):
        return calculate_architecture_parameters(
            base_params=req.base_parameters,
            d_model=req.d_model,
            num_layers=req.num_layers,
            num_experts=req.num_experts,
            expert_dim=req.expert_dim,
            lora_rank=req.lora_rank
        )

    @app.post("/api/chat")
    async def chat_endpoint(req: ChatRequest):
        model_instance = wrap(req.model)
        response = model_instance.generate(
            req.prompt,
            use_deliberation=bool(req.deliberate),
            use_agent=bool(req.enable_agent)
        )
        return {"response": response, "model": req.model}

    @app.get("/api/dataset/preview")
    async def dataset_preview(path: str = "data/lunaite_training_data.jsonl"):
        if not os.path.exists(path):
            generate_preset_dataset(path)
        samples = load_dataset_file(path)
        return {"count": len(samples), "samples": samples[:20]}

    @app.websocket("/ws/telemetry")
    async def ws_telemetry(ws: WebSocket):
        await ws.accept()
        studio_state.register_ws(ws)
        try:
            while True:
                # Send live system telemetry every 2 seconds
                telem = get_system_telemetry()
                await ws.send_json({"type": "telemetry", "data": telem})
                await asyncio.sleep(2.0)
        except WebSocketDisconnect:
            studio_state.unregister_ws(ws)
        except Exception:
            studio_state.unregister_ws(ws)

    # Static web app mount
    if os.path.exists(WEB_DIR):
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/")
    async def index():
        index_file = os.path.join(WEB_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return HTMLResponse("<h1>Lunaite Architecture Studio</h1><p>Web interface not found.</p>")


def launch_studio(port: int = 8000, host: str = "127.0.0.1", open_browser: bool = True):
    """Launch the Lunaite Studio FastAPI web server."""
    if not HAS_FASTAPI:
        raise ImportError("FastAPI and Uvicorn are required. Install with: pip install fastapi uvicorn")
    import uvicorn
    import webbrowser

    url = f"http://{host}:{port}"
    print(f"[*] Starting Lunaite Architecture Studio on {url}")
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port)
