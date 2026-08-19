"""
Lunaite AI 10B — Interactive Studio Backend Server
===================================================
High-performance FastAPI server powering the 1-Click Lunaite 10B Training Studio,
providing dataset management, dynamic parameter calculation, live WebSocket telemetry,
asynchronous model fine-tuning, and interactive streaming chat.
"""

import os
import sys
import json
import time
import asyncio
import subprocess
import threading
from typing import Dict, Any, List, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import psutil
import torch
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from lunaite.core.architecture import calculate_architecture_parameters
from lunaite_apps import (
    execute_app_intent,
    get_system_telemetry,
    take_screenshot,
    read_clipboard,
    write_clipboard,
    run_powershell,
    read_file_content,
    write_file_content,
    kill_process,
    open_url_in_browser,
    list_directory,
)
from lunaite_agent import detect_app_intent, execute_app_action

class AppActionRequest(BaseModel):
    intent: Optional[str] = None
    action_str: Optional[str] = None
    params: Optional[Dict[str, Any]] = None

# ─── Initialization ───────────────────────────────────────────────────────────

app = FastAPI(
    title="Lunaite AI 10B Studio Server",
    description="Backend API and WebSocket stream for Lunaite 10B Model Architecture & Training Studio",
    version="2.0.0"
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

KNOWLEDGE_GRAPH_FILE = os.path.join(DATA_DIR, "knowledge_graph.json")

# ─── In-Memory Knowledge Graph ────────────────────────────────────────────────

def _load_knowledge_graph() -> Dict[str, Any]:
    if os.path.exists(KNOWLEDGE_GRAPH_FILE):
        try:
            with open(KNOWLEDGE_GRAPH_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"nodes": [], "links": []}

def _save_knowledge_graph(graph: Dict[str, Any]):
    try:
        with open(KNOWLEDGE_GRAPH_FILE, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2)
    except Exception:
        pass

knowledge_graph = _load_knowledge_graph()

# ─── Training State Manager ───────────────────────────────────────────────────

class TrainingState:
    def __init__(self):
        self.status = "idle"  # idle, training, merging, completed, error, stopped
        self.process: Optional[subprocess.Popen] = None
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_step = 0
        self.total_steps = 0
        self.progress = 0.0
        self.current_loss = 0.0
        self.identity_loss = 0.0
        self.learning_rate = 0.0
        self.loss_history: List[Dict[str, Any]] = []
        self.logs: List[str] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.error_message: Optional[str] = None
        self.model_metadata: Dict[str, Any] = {}
        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            self.status = "idle"
            self.process = None
            self.current_epoch = 0
            self.total_epochs = 0
            self.current_step = 0
            self.total_steps = 0
            self.progress = 0.0
            self.current_loss = 0.0
            self.identity_loss = 0.0
            self.learning_rate = 0.0
            self.loss_history = []
            self.logs = []
            self.start_time = None
            self.end_time = None
            self.error_message = None
            self.model_metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "status": self.status,
                "current_epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "progress": self.progress,
                "current_loss": self.current_loss,
                "identity_loss": self.identity_loss,
                "learning_rate": self.learning_rate,
                "loss_history": self.loss_history[-100:],  # last 100 points
                "logs": self.logs[-200:],                  # last 200 logs
                "start_time": self.start_time,
                "elapsed": round(time.time() - self.start_time, 1) if self.start_time and self.status == "training" else (round(self.end_time - self.start_time, 1) if self.start_time and self.end_time else 0.0),
                "error_message": self.error_message,
                "model_metadata": self.model_metadata
            }

training_state = TrainingState()

# ─── WebSocket Connection Manager ─────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()

def broadcast_sync(payload: Dict[str, Any]):
    """Helper to broadcast WebSocket events from background threads."""
    asyncio.run(manager.broadcast(payload))


# ─── Background Training Runner ───────────────────────────────────────────────

def run_training_worker(config: Dict[str, Any]):
    global training_state
    
    with training_state.lock:
        training_state.status = "training"
        training_state.start_time = time.time()
        training_state.total_epochs = config.get("epochs", 5)
        training_state.logs.append(f"[*] Initializing Lunaite AI 10B training engine on {config.get('dataset_path')}...")

    # Build command line
    cmd = [
        sys.executable,
        "train_lunaite_lora.py",
        "--model-id", config.get("model_id", "Qwen/Qwen2.5-7B"),
        "--dataset", config.get("dataset_path", "data/lunaite_training_data.jsonl"),
        "--epochs", str(config.get("epochs", 5)),
        "--batch-size", str(config.get("batch_size", 1)),
        "--grad-accum", str(config.get("grad_accum", 8)),
        "--lr", str(config.get("lr", 2e-4)),
        "--rank", str(config.get("rank", 64)),
        "--alpha", str(config.get("alpha", 128)),
        "--identity-weight", str(config.get("identity_weight", 3.0)),
        "--quantization", config.get("quantization", "none"),
        "--emit-json"
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.path.abspath(".")
        )
        training_state.process = proc

        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            raw_line = line.strip()
            if not raw_line:
                continue

            if raw_line.startswith("__LUNAITE_EVENT__:"):
                try:
                    event_data = json.loads(raw_line[len("__LUNAITE_EVENT__:"):])
                    ev_type = event_data.get("event")

                    with training_state.lock:
                        if ev_type == "status":
                            msg = event_data.get("message", "")
                            training_state.logs.append(f"[*] {msg}")
                        elif ev_type == "step":
                            training_state.current_step = event_data.get("step", 0)
                            training_state.total_steps = event_data.get("total_steps", 0)
                            training_state.current_loss = event_data.get("loss", 0.0)
                            training_state.learning_rate = event_data.get("lr", 0.0)
                            training_state.progress = event_data.get("progress", 0.0)
                            training_state.loss_history.append({
                                "step": training_state.current_step,
                                "loss": event_data.get("loss", 0.0),
                                "epoch": event_data.get("epoch", 1),
                                "lr": event_data.get("lr", 0.0)
                            })
                        elif ev_type == "epoch_end":
                            training_state.current_epoch = event_data.get("epoch", 0)
                            training_state.identity_loss = event_data.get("identity_loss", 0.0)
                            training_state.logs.append(
                                f"[+] Epoch {training_state.current_epoch}/{training_state.total_epochs} complete. "
                                f"Avg Loss: {event_data.get('avg_loss'):.4f}, Identity Loss: {event_data.get('identity_loss'):.4f}"
                            )
                        elif ev_type == "train_complete":
                            training_state.model_metadata = event_data.get("metadata", {})
                            training_state.logs.append("[SUCCESS] Lunaite AI 10B training and merge complete!")

                    # Broadcast event
                    try:
                        asyncio.run(manager.broadcast({"type": "telemetry", "state": training_state.to_dict(), "event": event_data}))
                    except Exception:
                        pass

                except Exception as ex:
                    with training_state.lock:
                        training_state.logs.append(f"[ERR] JSON parse: {ex}")
            else:
                with training_state.lock:
                    training_state.logs.append(raw_line)
                try:
                    asyncio.run(manager.broadcast({"type": "log", "line": raw_line}))
                except Exception:
                    pass

        proc.wait()
        with training_state.lock:
            training_state.end_time = time.time()
            if proc.returncode == 0:
                training_state.status = "completed"
                training_state.progress = 100.0
            else:
                if training_state.status != "stopped":
                    training_state.status = "error"
                    training_state.error_message = f"Training exited with code {proc.returncode}"

    except Exception as e:
        with training_state.lock:
            training_state.status = "error"
            training_state.error_message = str(e)
            training_state.logs.append(f"[FATAL] {str(e)}")
            training_state.end_time = time.time()

    finally:
        try:
            asyncio.run(manager.broadcast({"type": "status_change", "state": training_state.to_dict()}))
        except Exception:
            pass


# ─── API Models ───────────────────────────────────────────────────────────────

class StartTrainingRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str = "Qwen/Qwen2.5-7B"
    dataset_path: str = "data/lunaite_training_data.jsonl"
    epochs: int = 5
    batch_size: int = 1
    grad_accum: int = 8
    lr: float = 2e-4
    rank: int = 64
    alpha: int = 128
    identity_weight: float = 3.0
    quantization: str = "none"

class CustomSample(BaseModel):
    instruction: str
    output: str

class CustomDatasetRequest(BaseModel):
    filename: str = "custom_lunaite_dataset.jsonl"
    samples: List[CustomSample]

class ChatRequest(BaseModel):
    model: str = "lunaite-ai"
    prompt: str
    system_prompt: Optional[str] = "You are Lunaite AI, a 10B-parameter intelligence created by Swasthik Shetty."
    temperature: float = 0.7


# ─── REST Endpoints ───────────────────────────────────────────────────────────

@app.get("/api/system/status")
async def get_system_status():
    """Return hardware resources, GPU availability, and Ollama status."""
    cuda_avail = torch.cuda.is_available()
    gpu_info = None
    if cuda_avail:
        gpu_info = {
            "device_name": torch.cuda.get_device_name(0),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            "vram_allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2),
            "bf16_supported": torch.cuda.is_bf16_supported()
        }

    cpu_percent = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()

    # Check Ollama
    ollama_running = False
    ollama_models = []
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1.5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                ollama_running = True
                ollama_models = [m.get("name") for m in data.get("models", [])]
    except Exception:
        ollama_running = False

    return {
        "cuda_available": cuda_avail,
        "gpu": gpu_info,
        "cpu_usage_percent": cpu_percent,
        "ram_total_gb": round(ram.total / (1024**3), 2),
        "ram_used_gb": round(ram.used / (1024**3), 2),
        "ram_percent": ram.percent,
        "ollama_running": ollama_running,
        "ollama_models": ollama_models
    }


@app.get("/api/architecture/calculate")
async def get_architecture_calculation(
    base_params: int = 8_200_000_000,
    target_params: int = 10_000_000_000,
    d_model: int = 4096,
    num_layers: int = 36
):
    """Calculate 10B architecture projection matrices and rank values."""
    return calculate_10b_architecture(
        base_params=base_params,
        target_params=target_params,
        d_model=d_model,
        num_layers=num_layers
    )


@app.get("/api/dataset/preview")
async def preview_dataset(path: str = "data/lunaite_training_data.jsonl", limit: int = 20):
    """Preview dataset samples, sample count, and identity coverage."""
    if not os.path.isabs(path):
        path = os.path.abspath(path)

    if not os.path.exists(path):
        return {"exists": False, "samples": [], "total_count": 0, "identity_count": 0}

    samples = []
    identity_count = 0
    total_count = 0
    identity_keywords = ["who are you", "what are you", "your name", "company", "made you", "chatgpt", "claude", "gemini", "qwen", "swasthik", "lunaite"]

    if path.endswith(".md") or path.endswith(".markdown") or path.endswith(".txt"):
        from train_lunaite_lora import parse_markdown_to_samples
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw_samples = parse_markdown_to_samples(f.read())
            total_count = len(raw_samples)
            for idx, item in enumerate(raw_samples):
                instr = item["instruction"]
                out = item["output"]
                is_ident = any(kw in instr.lower() for kw in identity_keywords)
                if is_ident:
                    identity_count += 1
                if len(samples) < limit:
                    samples.append({
                        "index": idx + 1,
                        "instruction": instr,
                        "output": out,
                        "is_identity": is_ident,
                        "length": len(instr) + len(out)
                    })
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    instr = item.get("instruction") or item.get("prompt") or item.get("input") or ""
                    out = item.get("output") or item.get("response") or item.get("answer") or ""
                    is_ident = any(kw in instr.lower() for kw in identity_keywords)
                    if is_ident:
                        identity_count += 1
                    total_count += 1
                    if len(samples) < limit:
                        samples.append({
                            "index": total_count,
                            "instruction": instr,
                            "output": out,
                            "is_identity": is_ident,
                            "length": len(instr) + len(out)
                        })
                except Exception:
                    continue

    return {
        "exists": True,
        "path": path,
        "filename": os.path.basename(path),
        "total_count": total_count,
        "identity_count": identity_count,
        "identity_ratio": round(identity_count / max(total_count, 1) * 100, 1),
        "samples": samples
    }


@app.post("/api/dataset/upload")
async def upload_dataset_file(file: UploadFile = File(...)):
    """Upload custom dataset file (.jsonl, .json, .csv, .md, .txt)."""
    filename = file.filename
    clean_name = "".join(c for c in filename if c.isalnum() or c in "._- ")
    target_path = os.path.join(DATA_DIR, clean_name)

    content = await file.read()
    
    # Process & convert if needed
    if clean_name.endswith(".csv"):
        import csv
        import io
        csv_text = content.decode("utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(csv_text))
        jsonl_path = os.path.join(DATA_DIR, f"{os.path.splitext(clean_name)[0]}.jsonl")
        count = 0
        with open(jsonl_path, "w", encoding="utf-8") as out_f:
            for row in reader:
                instr = row.get("instruction") or row.get("prompt") or row.get("input") or row.get("question") or ""
                resp = row.get("output") or row.get("response") or row.get("answer") or ""
                if instr and resp:
                    out_f.write(json.dumps({"instruction": instr, "output": resp}) + "\n")
                    count += 1
        target_path = jsonl_path
    elif clean_name.endswith(".md") or clean_name.endswith(".markdown"):
        from train_lunaite_lora import parse_markdown_to_samples
        text = content.decode("utf-8", errors="ignore")
        parsed = parse_markdown_to_samples(text)
        jsonl_path = os.path.join(DATA_DIR, f"{os.path.splitext(clean_name)[0]}.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as out_f:
            for s in parsed:
                out_f.write(json.dumps(s) + "\n")
        target_path = jsonl_path
    else:
        with open(target_path, "wb") as f:
            f.write(content)

    return {
        "status": "success",
        "filename": os.path.basename(target_path),
        "saved_path": target_path
    }


@app.post("/api/dataset/create-custom")
async def create_custom_dataset(payload: CustomDatasetRequest):
    """Create a new dataset from user submitted prompt/response pairs."""
    clean_name = "".join(c for c in payload.filename if c.isalnum() or c in "._- ")
    if not clean_name.endswith(".jsonl"):
        clean_name += ".jsonl"
    target_path = os.path.join(DATA_DIR, clean_name)

    with open(target_path, "w", encoding="utf-8") as f:
        for sample in payload.samples:
            if sample.instruction.strip() and sample.output.strip():
                f.write(json.dumps({"instruction": sample.instruction.strip(), "output": sample.output.strip()}) + "\n")

    return {
        "status": "success",
        "filename": clean_name,
        "saved_path": target_path,
        "samples_count": len(payload.samples)
    }


@app.post("/api/dataset/generate-preset")
async def generate_preset_dataset():
    """Trigger the multi-disciplinary dataset generator (generate_dataset.py)."""
    proc = subprocess.run([sys.executable, "generate_dataset.py"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Failed to generate dataset: {proc.stderr}")
    
    return {
        "status": "success",
        "message": "Generated 80+ multi-domain identity-enforced training samples",
        "dataset_path": "data/lunaite_training_data.jsonl"
    }


@app.post("/api/train/start")
async def start_training(config: StartTrainingRequest):
    """Start 10B training asynchronously."""
    global training_state
    if training_state.status == "training":
        raise HTTPException(status_code=400, detail="Training is already in progress!")

    training_state.reset()
    
    # Launch worker in background thread
    t = threading.Thread(target=run_training_worker, args=(config.dict(),), daemon=True)
    t.start()

    return {
        "status": "started",
        "message": "Lunaite AI 10B training pipeline initialized!",
        "config": config.dict()
    }


@app.post("/api/train/stop")
async def stop_training():
    """Stop active training process."""
    global training_state
    if training_state.process and training_state.status == "training":
        try:
            training_state.process.terminate()
            with training_state.lock:
                training_state.status = "stopped"
                training_state.logs.append("[!] Training stopped by user.")
                training_state.end_time = time.time()
            return {"status": "stopped", "message": "Training process terminated."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return {"status": "idle", "message": "No active training process."}


@app.get("/api/train/status")
async def get_training_status():
    """Fetch current training telemetry."""
    return training_state.to_dict()


@app.get("/api/status")
@app.get("/api/system/status")
async def get_system_diagnostics():
    """Fetch live hardware and Ollama server status."""
    gpu_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_avail else "RTX 5070 Laptop GPU"
    
    ollama_ok = False
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                ollama_ok = True
    except Exception:
        ollama_ok = False

    return {
        "status": "online",
        "model": "lunaite-ai:latest (27B Frontier)",
        "gpu_available": gpu_avail,
        "gpu_device": gpu_name,
        "ollama_online": ollama_ok,
        "creator": "Swasthik Shetty"
    }


@app.post("/api/export/ollama")
async def export_to_ollama(model_name: str = "lunaite-ai", modelfile_path: str = "Modelfile"):
    """Rebuild model into Ollama from Modelfile."""
    try:
        proc = subprocess.run(["ollama", "create", model_name, "-f", modelfile_path], capture_output=True, text=True)
        if proc.returncode == 0:
            return {"status": "success", "message": f"Model '{model_name}' successfully created in Ollama!", "output": proc.stdout}
        else:
            return {"status": "error", "message": "Ollama create returned an error", "error": proc.stderr}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama execution error: {str(e)}")


@app.post("/api/chat/generate")
async def chat_generate(req: ChatRequest):
    """Interactive chat with Lunaite AI via Ollama API (non-streaming fallback)."""
    import urllib.request
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": req.model,
        "prompt": req.prompt,
        "system": req.system_prompt,
        "stream": False,
        "options": {
            "temperature": req.temperature,
            "num_predict": 512
        }
    }
    
    try:
        data_bytes = json.dumps(payload).encode("utf-8")
        http_req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(http_req, timeout=90) as resp:
            result = json.loads(resp.read().decode())
            return {
                "response": result.get("response", ""),
                "total_duration": result.get("total_duration", 0),
                "eval_count": result.get("eval_count", 0),
                "model": req.model
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to communicate with Ollama: {str(e)}")


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """Token-streaming SSE chat endpoint — renders each word the instant Ollama produces it."""
    import urllib.request

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": req.model,
        "prompt": req.prompt,
        "system": req.system_prompt,
        "stream": True,
        "options": {
            "temperature": req.temperature,
            "num_predict": 512
        }
    }

    async def token_generator():
        try:
            data_bytes = json.dumps(payload).encode("utf-8")
            http_req = urllib.request.Request(
                url, data=data_bytes,
                headers={"Content-Type": "application/json"}
            )
            # Run blocking urllib call in thread pool so we don't block the event loop
            loop = asyncio.get_event_loop()

            def read_stream():
                chunks = []
                with urllib.request.urlopen(http_req, timeout=90) as resp:
                    for raw_line in resp:
                        line = raw_line.strip()
                        if line:
                            chunks.append(line)
                return chunks

            # Stream token-by-token using httpx async for real non-blocking streaming
            import httpx
            async with httpx.AsyncClient(timeout=90) as client:
                async with client.stream(
                    "POST", url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            token = chunk.get("response", "")
                            done = chunk.get("done", False)
                            if token:
                                # SSE format: data: <json>\n\n
                                yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
                            if done:
                                yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
                                return
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


@app.post("/api/app/action")
async def handle_app_action(req: AppActionRequest):
    """Execute Windows desktop app control from web interface."""
    try:
        if req.action_str:
            res = execute_app_action(req.action_str)
            return {"status": "success", "result": res}
        elif req.intent:
            params = req.params or {}
            res = execute_app_intent(req.intent, **params)
            return {"status": "success", "result": res}
        return {"status": "error", "message": "No intent or action_str provided"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── JARVIS Deep System Access Endpoints ──────────────────────────────────────

@app.get("/api/system/telemetry")
async def system_telemetry():
    """Return live CPU, RAM, disk, battery, uptime telemetry."""
    try:
        data = get_system_telemetry()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ScreenshotRequest(BaseModel):
    save_dir: str = "."

@app.post("/api/system/screenshot")
async def system_screenshot(req: ScreenshotRequest = ScreenshotRequest()):
    """Capture a full screenshot and return base64 thumbnail + file path."""
    try:
        result = take_screenshot(req.save_dir)
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ClipboardRequest(BaseModel):
    action: str = "read"   # "read" or "write"
    text: Optional[str] = None

@app.post("/api/system/clipboard")
async def system_clipboard(req: ClipboardRequest):
    """Read or write the Windows clipboard."""
    try:
        if req.action == "write":
            result = write_clipboard(req.text or "")
        else:
            result = read_clipboard()
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PowerShellRequest(BaseModel):
    command: str
    timeout: int = 20

@app.post("/api/system/run")
async def system_run_powershell(req: PowerShellRequest):
    """Execute an arbitrary PowerShell command and return its output."""
    try:
        output = run_powershell(req.command, req.timeout)
        return {"output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class FileReadRequest(BaseModel):
    path: str
    max_lines: int = 100

@app.post("/api/system/file/read")
async def system_file_read(req: FileReadRequest):
    """Read content of a local file."""
    content = read_file_content(req.path, req.max_lines)
    return {"content": content}


class FileWriteRequest(BaseModel):
    path: str
    content: str

@app.post("/api/system/file/write")
async def system_file_write(req: FileWriteRequest):
    """Write content to a local file."""
    result = write_file_content(req.path, req.content)
    return {"result": result}


# ─── Knowledge Graph Endpoints ─────────────────────────────────────────────────

class KnowledgeNodeRequest(BaseModel):
    id: str
    label: str
    type: str = "concept"   # person | concept | place | tool | event
    source_id: Optional[str] = None   # link from this node
    link_label: Optional[str] = "related_to"

@app.get("/api/knowledge/graph")
async def get_knowledge_graph():
    """Return the full knowledge graph (nodes + links)."""
    return knowledge_graph

@app.post("/api/knowledge/add")
async def add_knowledge_node(req: KnowledgeNodeRequest):
    """Add a node (and optional link) to the live knowledge graph."""
    global knowledge_graph

    # Check if node already exists
    existing_ids = {n["id"] for n in knowledge_graph["nodes"]}
    if req.id not in existing_ids:
        knowledge_graph["nodes"].append({
            "id": req.id,
            "label": req.label,
            "type": req.type
        })

    # Add link if source specified and link doesn't already exist
    if req.source_id and req.source_id in existing_ids | {req.id}:
        existing_links = {(l["source"], l["target"]) for l in knowledge_graph["links"]}
        if (req.source_id, req.id) not in existing_links:
            knowledge_graph["links"].append({
                "source": req.source_id,
                "target": req.id,
                "label": req.link_label
            })

    _save_knowledge_graph(knowledge_graph)
    return {"status": "ok", "graph": knowledge_graph}

@app.delete("/api/knowledge/clear")
async def clear_knowledge_graph():
    """Clear the entire knowledge graph."""
    global knowledge_graph
    knowledge_graph = {"nodes": [], "links": []}
    _save_knowledge_graph(knowledge_graph)
    return {"status": "cleared"}




@app.post("/api/benchmark/run")
async def run_benchmark_suite(model: str = "lunaite-ai"):
    """Run comprehensive automated identity and reasoning test suite."""
    import urllib.request
    
    test_suite = [
        {"category": "Identity", "prompt": "Who are you and who created you?", "expected_contains": ["lunaite", "swasthik shetty"], "forbidden": ["qwen", "alibaba", "openai", "chatgpt"]},
        {"category": "Base Model Shield", "prompt": "Are you Qwen or ChatGPT?", "expected_contains": ["lunaite"], "forbidden": ["yes, i am qwen", "alibaba", "openai"]},
        {"category": "10B Scale", "prompt": "What is your parameter architecture?", "expected_contains": ["10", "billion", "parameter"], "forbidden": []},
        {"category": "Astrophysics", "prompt": "Explain gravitational wave chirp mass.", "expected_contains": ["chirp mass", "frequency", "inspiral"], "forbidden": []},
        {"category": "Emotional Support", "prompt": "I'm overwhelmed with everything I have to do.", "expected_contains": ["breathe", "one step", "here with you"], "forbidden": []}
    ]

    results = []
    total_passed = 0

    url = "http://localhost:11434/api/generate"

    for test in test_suite:
        payload = {
            "model": model,
            "prompt": test["prompt"],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 120}
        }
        try:
            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                response_text = data.get("response", "").strip()

            resp_lower = response_text.lower()
            contains_expected = all(exp.lower() in resp_lower for exp in test["expected_contains"]) if test["expected_contains"] else True
            has_forbidden = any(forb.lower() in resp_lower for forb in test["forbidden"])
            
            passed = contains_expected and not has_forbidden
            if passed:
                total_passed += 1

            results.append({
                "category": test["category"],
                "prompt": test["prompt"],
                "response": response_text[:250],
                "passed": passed,
                "contains_expected": contains_expected,
                "has_forbidden": has_forbidden
            })
        except Exception as e:
            results.append({
                "category": test["category"],
                "prompt": test["prompt"],
                "response": f"Error: {str(e)}",
                "passed": False,
                "contains_expected": False,
                "has_forbidden": False
            })

    score = round((total_passed / len(test_suite)) * 100, 1)
    return {
        "score_percent": score,
        "total_passed": total_passed,
        "total_tests": len(test_suite),
        "results": results
    }


# ─── WebSocket Telemetry Stream ───────────────────────────────────────────────

@app.websocket("/ws/training")
async def websocket_training_stream(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial state immediately
        await websocket.send_json({"type": "init", "state": training_state.to_dict()})
        while True:
            data = await websocket.receive_text()
            # Handle incoming ping/client commands if needed
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


# ─── Static Frontend Delivery ─────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

@app.get("/")
async def serve_index():
    index_file = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h1>Lunaite AI 10B Studio is starting...</h1>")


if __name__ == "__main__":
    import uvicorn
    print("================================================================")
    print("  * LUNAITE AI 10B -- Interactive Dataset & Training Studio")
    print("  Created by Swasthik Shetty")
    print("  Web Studio running at: http://localhost:8000")
    print("================================================================")
    uvicorn.run("lunaite_studio_server:app", host="0.0.0.0", port=8000, reload=False)
