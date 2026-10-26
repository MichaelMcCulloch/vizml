import asyncio
import json
import tempfile
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, ValidationError

from .builder import compile_meta, export_python
from .graph import analyze
from .schema import Graph
from . import checkpoints

app = FastAPI(title="vizml", version="0.1.0")


@app.get("/api/health")
async def health():
    return {"status": "ok", "engine": "pytorch", "protocol": 1}


@app.post("/api/analyze")
async def analyze_graph(graph: Graph):
    return analyze(graph)


@app.get("/api/templates/catan_convnext_small/encoder.py")
async def catan_encoder_source():
    return FileResponse(Path(__file__).with_name("catan_convnext.py"), media_type="text/x-python", filename="catan_convnext_encoder.py")


class CheckpointExport(BaseModel):
    graph: Graph
    mode: Literal["architecture", "checkpoint", "state_dict"] = "architecture"


@app.post("/api/checkpoints/import")
async def import_checkpoint(request: Request, filename: str = "checkpoint.pt"):
    if int(request.headers.get("content-length", "0")) > checkpoints.MAX_UPLOAD:
        raise HTTPException(413, "Checkpoint imports are limited to 2 GiB.")
    temporary = tempfile.NamedTemporaryFile(dir=checkpoints.cache_dir(), suffix=".upload", delete=False)
    path = Path(temporary.name)
    try:
        size = 0
        with temporary:
            async for chunk in request.stream():
                size += len(chunk)
                if size > checkpoints.MAX_UPLOAD:
                    raise HTTPException(413, "Checkpoint imports are limited to 2 GiB.")
                temporary.write(chunk)
        return await asyncio.to_thread(checkpoints.register_file, path, filename)
    except (ValueError, RuntimeError, OSError) as error:
        raise HTTPException(400, str(error)[:800]) from error
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/checkpoints/export")
async def export_checkpoint(request: CheckpointExport):
    try:
        return await asyncio.to_thread(checkpoints.export_checkpoint, request.graph, request.mode)
    except (ValueError, RuntimeError, OSError) as error:
        raise HTTPException(400, str(error)[:800]) from error


@app.get("/api/checkpoints/{key}")
async def checkpoint_info(key: str):
    try:
        return checkpoints.manifest(key)
    except ValueError as error:
        raise HTTPException(404, str(error)) from error


@app.post("/api/checkpoints/{key}/bind")
async def bind_checkpoint(key: str, graph: Graph):
    try:
        return await asyncio.to_thread(checkpoints.bind_checkpoint, key, graph)
    except (ValueError, RuntimeError, OSError) as error:
        raise HTTPException(400, str(error)[:800]) from error


@app.get("/api/checkpoints/{key}/download")
async def download_checkpoint(key: str):
    try:
        info = checkpoints.manifest(key)
        path = checkpoints.cache_path(info.get("source_id", key))
        if not path.is_file():
            raise ValueError("Checkpoint file is unavailable. Import it again.")
        return FileResponse(path, media_type="application/octet-stream", filename=info["filename"])
    except ValueError as error:
        raise HTTPException(404, str(error)) from error


@app.websocket("/ws")
async def graph_events(socket: WebSocket):
    await socket.accept()
    await socket.send_json({"type": "connected", "protocol": 1})
    try:
        while True:
            payload = await socket.receive_text()
            request_id = None
            try:
                if len(payload) > 2_000_000:
                    raise ValueError("This graph exceeds the 2 MB event limit.")
                event = json.loads(payload)
                if not isinstance(event, dict):
                    raise ValueError("An event must be a JSON object.")
                request_id = event.get("request_id")
                if event.get("type") == "ping":
                    await socket.send_json({"type": "pong", "request_id": request_id})
                    continue
                graph = Graph.model_validate(event.get("graph"))
                if event.get("type") == "graph.changed":
                    response = {"type": "graph.analyzed", "analysis": analyze(graph)}
                elif event.get("type") == "model.export":
                    response = {"type": "model.exported", "source": export_python(graph)}
                elif event.get("type") == "model.compile":
                    response = {"type": "model.compiled", "result": await asyncio.to_thread(compile_meta, graph)}
                else:
                    raise ValueError("Unknown event type.")
                await socket.send_json({**response, "request_id": request_id})
            except (ValidationError, ValueError, RuntimeError, TypeError, KeyError, OverflowError, AssertionError) as error:
                message = "Invalid graph data. Check node types and input values." if isinstance(error, ValidationError) else str(error)
                await socket.send_json({"type": "error", "request_id": request_id, "message": message[:800]})
    except WebSocketDisconnect:
        pass


@app.get("/")
async def frontend():
    return RedirectResponse(os.environ.get("VIZML_APP_URL", "http://127.0.0.1:5173/"))
