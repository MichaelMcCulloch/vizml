"""Persist explicit workspace saves independently of browser autosave."""
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from .schema import Graph

MAX_BYTES = 2_000_000


def root() -> Path:
    return Path(os.environ.get("VIZML_DATA_DIR", Path(__file__).resolve().parents[1] / ".vizml"))


def _path(kind: str, key: str) -> Path:
    if kind not in {"projects", "templates"}:
        raise ValueError("Unknown workspace collection.")
    try:
        canonical = str(UUID(key))
    except (ValueError, TypeError, AttributeError):
        raise ValueError("Invalid workspace ID.") from None
    if key != canonical:
        raise ValueError("Invalid workspace ID.")
    return root() / kind / f"{key}.json"


def get(kind: str, key: str) -> dict:
    try:
        return json.loads(_path(kind, key).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise LookupError("Saved item not found.") from None


def list_items(kind: str) -> list[dict]:
    directory = root() / kind
    if kind not in {"projects", "templates"}:
        raise ValueError("Unknown workspace collection.")
    if not directory.exists():
        return []
    items = []
    for path in directory.glob("*.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            if str(UUID(item["id"])) != path.stem:
                continue
            items.append({key: item[key] for key in ("id", "name", "updated_at")})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(items, key=lambda item: item["updated_at"], reverse=True)


def save(kind: str, name: str, graph: Graph, key: str | None = None) -> dict:
    name = name.strip()
    if not name or len(name) > 100:
        raise ValueError("Choose a name of 1 to 100 characters.")
    updating = key is not None
    key = key or str(uuid4())
    path = _path(kind, key)
    if kind == "templates" and path.exists():
        raise ValueError("Templates are saved as new copies.")
    if updating and not path.exists():
        raise LookupError("Saved item not found.")
    item = {"id": key, "name": name, "updated_at": datetime.now(timezone.utc).isoformat(), "graph": graph.model_dump()}
    payload = json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_BYTES:
        raise ValueError("Workspace item exceeds the 2 MB size limit.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".save-", delete=False) as file:
            temp_name = file.name
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
    return item


def delete(kind: str, key: str) -> None:
    path = _path(kind, key)
    try:
        path.unlink()
    except FileNotFoundError:
        raise LookupError("Saved item not found.") from None
