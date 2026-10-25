"""Local .pt interchange. Uploaded pickle classes are never executed/allowlisted."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import shutil
import uuid
import zipfile
from pathlib import Path

from .builder import export_python
from .graph import analyze
from .schema import Graph

MAX_UPLOAD = 2 * 1024**3
MAX_INITIALIZE = 512 * 1024**2
MAX_TENSORS = 30000


def cache_dir() -> Path:
    path = Path(os.environ.get("VIZML_CHECKPOINT_DIR", Path(__file__).resolve().parents[1] / ".vizml" / "checkpoints"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_path(key: str, suffix=".pt") -> Path:
    if not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", key):
        raise ValueError("Invalid checkpoint ID.")
    return cache_dir() / (key + suffix)


def manifest(key: str) -> dict:
    try:
        return json.loads(cache_path(key, ".json").read_text())
    except FileNotFoundError as e:
        raise ValueError("This local checkpoint is unavailable. Import its .pt file again.") from e


def write_manifest(info: dict):
    cache_path(info["id"], ".json").write_text(json.dumps(info, ensure_ascii=False, allow_nan=False))


def signature(graph: Graph) -> str:
    nodes = [{"id": n.id, "kind": n.data.kind, "params": n.data.params,
              **({"output_shape": n.data.model_dump().get("output_shape"), "parameter_tensors": n.data.model_dump().get("parameter_tensors", [])} if graph.mode == "schematic" else {})} for n in graph.nodes]
    spec = {"mode": graph.mode, "nodes": nodes, "edges": [[e.source, e.target] for e in graph.edges]}
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _state(payload):
    import torch
    if not isinstance(payload, dict):
        raise ValueError("Expected a tensor state_dict or a dictionary checkpoint. Re-save the model's state_dict rather than its Python class.")
    for key in ("model_state_dict", "state_dict", "model", "weights", "net"):
        item = payload.get(key)
        if isinstance(item, dict) and item and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in item.items()):
            return item, key
    if payload and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in payload.items()):
        return payload, None
    if payload.get("format") == "vizml.checkpoint" and (payload.get("weights") == "none" or payload.get("model_state_dict") == {}):
        return {}, "model_state_dict"
    raise ValueError("No supported model state_dict was found. Supported keys: model_state_dict, state_dict, model, weights, net.")


def _load(path: Path, *, inspect=False):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    if path.stat().st_size > MAX_UPLOAD:
        raise ValueError("Checkpoint imports are limited to 2 GiB per file.")
    is_zip = zipfile.is_zipfile(path)
    if is_zip:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if any("/code/" in i.filename or i.filename.endswith("/constants.pkl") for i in entries):
                raise ValueError("TorchScript archives are not supported. Export a state_dict checkpoint for inspection.")
            if len(entries) > 100000 or sum(i.file_size for i in entries) > MAX_UPLOAD + 32 * 1024**2:
                raise ValueError("Checkpoint archive exceeds the supported size.")
            if any(i.filename.endswith("data.pkl") and i.file_size > 24 * 1024**2 for i in entries):
                raise ValueError("Checkpoint metadata exceeds 24 MiB.")
    try:
        if inspect:
            with FakeTensorMode():
                return torch.load(path, weights_only=True, map_location="cpu", mmap=is_zip)
        return torch.load(path, weights_only=True, map_location="cpu", mmap=is_zip)
    except (pickle.UnpicklingError, EOFError, RuntimeError, TypeError, KeyError, ValueError, IndexError) as e:
        raise ValueError("Could not read this checkpoint with PyTorch's weights-only loader. Save a plain state_dict or a dictionary of tensors and primitive metadata; serialized Python model classes are not supported.") from e


def inspect_file(path: Path, filename: str, key: str) -> dict:
    import torch
    payload = _load(path, inspect=True)
    state, state_key = _state(payload)
    if len(state) > MAX_TENSORS:
        raise ValueError(f"Checkpoint contains more than {MAX_TENSORS:,} tensors.")
    tensors = []
    for name, tensor in state.items():
        if len(name) > 1000 or tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError("Only named, dense, non-quantized tensors are supported in this importer.")
        tensors.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch."),
                        "elements": tensor.numel(), "bytes": tensor.numel() * tensor.element_size(), "meta": str(tensor.device) == "meta"})
    if sum(t["bytes"] for t in tensors if not t["meta"]) > MAX_UPLOAD:
        raise ValueError("The checkpoint's logical tensor storage exceeds 2 GiB.")
    graph = None
    if payload.get("format") == "vizml.checkpoint":
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported vizml checkpoint format version.")
        graph = Graph.model_validate(payload.get("graph"))
        if not analyze(graph)["valid"]:
            raise ValueError("The checkpoint contains an invalid architecture graph.")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    training = payload.get("original_checkpoint", payload)
    if not isinstance(training, dict):
        training = payload
    return {"id": key, "filename": Path(filename).name[:180], "sha256": digest, "file_bytes": path.stat().st_size,
            "state_key": state_key, "tensor_count": len(tensors), "tensor_bytes": sum(t["bytes"] for t in tensors),
            "tensor_elements": sum(t["elements"] for t in tensors), "has_weights": bool(tensors) and not any(t["meta"] for t in tensors),
            "architecture": training.get("architecture") if isinstance(training.get("architecture"), str) else None,
            "decisions": training.get("decisions") if isinstance(training.get("decisions"), int) else None,
            "has_optimizer": any(k in payload for k in ("optimizer", "optimizer_state_dict")) or (isinstance(payload.get("original_checkpoint"), dict) and any(k in payload["original_checkpoint"] for k in ("optimizer", "optimizer_state_dict"))),
            "graph": graph.model_dump() if graph else None, "tensors": tensors}


def register_file(path: Path, filename: str) -> dict:
    """Retain a local copy; the supplied original is never changed."""
    key = str(uuid.uuid4())
    info = inspect_file(path, filename, key)
    shutil.copyfile(path, cache_path(key))
    write_manifest(info)
    return info


def _model(graph: Graph, device="meta"):
    scope = {"__name__": "vizml_checkpoint_model"}
    exec(compile(export_python(graph), "<vizml generated model>", "exec"), scope)
    return scope["build_model"](device)


def expected_tensors(graph: Graph) -> list[dict]:
    if graph.mode == "schematic":
        return [{"name": t["name"], "shape": t["shape"], "node_id": node.id, "required": True}
                for node in graph.nodes for t in node.data.model_dump().get("parameter_tensors", [])]
    model = _model(graph)
    return [{"name": name, "shape": list(tensor.shape), "required": True} for name, tensor in model.state_dict().items()]


def compatibility(info: dict, graph: Graph) -> dict:
    analysis = analyze(graph)
    if not analysis["valid"]:
        raise ValueError("Resolve graph errors before pairing a checkpoint.")
    expected = expected_tensors(graph)
    source = {t["name"]: t for t in info["tensors"]}
    # Common training wrappers do not change tensor identity.
    aliases = {}
    for name in source:
        clean = name
        for _ in range(3):
            prefix = next((p for p in ("module.", "_orig_mod.", "model.") if clean.startswith(p)), None)
            if not prefix: break
            clean = clean[len(prefix):]
        aliases.setdefault(clean, name)
    missing, mismatches, mapping = [], [], {}
    for tensor in expected:
        key = tensor["name"] if tensor["name"] in source else aliases.get(tensor["name"])
        if key is None:
            missing.append(tensor["name"])
        elif source[key]["shape"] != tensor["shape"]:
            mismatches.append({"name": tensor["name"], "expected": tensor["shape"], "actual": source[key]["shape"]})
        else:
            mapping[tensor["name"]] = key
    claimed = graph.metadata.get("checkpoint", {})
    claimed_hash = claimed.get("sha256") if isinstance(claimed, dict) else None
    return {"compatible": bool(expected) and info["has_weights"] and not missing and not mismatches,
            "matched": len(mapping), "expected": len(expected), "missing": missing, "mismatches": mismatches,
            "mapping": mapping, "snapshot_match": info["sha256"] == claimed_hash if claimed_hash else None,
            "claimed_sha256": claimed_hash, "parameters": analysis["parameters"], "signature": signature(graph)}


def bind_checkpoint(key: str, graph: Graph) -> dict:
    info = manifest(key)
    report = compatibility(info, graph)
    if not report["compatible"]:
        detail = ", ".join(report["missing"][:3]) or ", ".join(m["name"] for m in report["mismatches"][:3])
        raise ValueError(f"Checkpoint does not match this architecture ({report['matched']}/{report['expected']} tensors). {detail}".strip())
    # A binding records the exact graph semantics; positions and labels may change.
    bound_id = str(uuid.uuid4())
    bound = {**info, "id": bound_id, "source_id": info.get("source_id", key), "binding": report,
             "graph": graph.model_dump()}
    write_manifest(bound)
    updated = graph.model_dump()
    updated["metadata"] = {**updated.get("metadata", {}), "weights": {"id": bound_id, "filename": info["filename"], "sha256": info["sha256"],
                            "matched": report["matched"], "expected": report["expected"], "snapshot_match": report["snapshot_match"], "signature": report["signature"]}}
    return {"graph": updated, "checkpoint": bound, "report": report}


def _clean_export_graph(graph: Graph):
    data = graph.model_dump()
    data["metadata"].pop("weights", None)  # a portable bundle never relies on a cache ID
    return data


def export_checkpoint(graph: Graph, mode: str) -> dict:
    import torch
    if mode not in {"architecture", "checkpoint", "state_dict"}:
        raise ValueError("Unknown checkpoint export mode.")
    analysis = analyze(graph)
    if not analysis["valid"]:
        raise ValueError("Resolve graph errors before exporting.")
    state, original, origin = None, None, "none"
    if mode != "architecture":
        weights = graph.metadata.get("weights", {})
        key = weights.get("id") if isinstance(weights, dict) else None
        if key:
            info = manifest(key)
            if info.get("binding", {}).get("signature") != signature(graph):
                raise ValueError("Architecture semantics changed since these weights were paired. Re-import/pair the checkpoint before exporting weights.")
            payload = _load(cache_path(info.get("source_id", key)))
            source_state, _ = _state(payload)
            if graph.mode == "schematic":
                state = source_state  # retain non-parameter buffers and the native module names
            else:
                state = {target: source_state[source] for target, source in info["binding"]["mapping"].items()}
            original = payload.get("original_checkpoint", payload) if payload.get("format") == "vizml.checkpoint" else payload
            origin = "imported"
        else:
            if not analysis.get("executable", True):
                raise ValueError("Pair the original .pt weights to this descriptive graph, or export architecture only.")
            meta = _model(graph)
            storage = sum(t.numel() * t.element_size() for t in meta.state_dict().values())
            if storage > MAX_INITIALIZE:
                raise ValueError("Initializing this model exceeds the 512 MiB weight limit. Export architecture only, or import existing weights.")
            state = _model(graph, "cpu").state_dict()
            origin = "initialized"
    data = state if mode == "state_dict" else {"format": "vizml.checkpoint", "format_version": 1,
            "graph": _clean_export_graph(graph), "model_state_dict": state, "weights": origin,
            **({"original_checkpoint": original} if original is not None else {})}
    key = str(uuid.uuid4()); path = cache_path(key)
    try:
        torch.save(data, path)
        info = inspect_file(path, f"{graph.name[:100]}.pt", key)
        info["weights_origin"] = origin
        write_manifest(info)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return {**info, "download_url": f"/api/checkpoints/{key}/download"}
