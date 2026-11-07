"""Results format shared by every device (Mac, Windows iGPU, Windows dGPU).

Each notebook run writes ONE file: ``experiments/<axis>/results/<device>.json``.
It holds every table the run produced plus the run's config, prompts and machine,
so each device commits only its own file (no merge conflicts) and ``figures.py``
can put all devices on the same graph.

This module needs only the standard library and pandas: the Windows (PyTorch)
notebooks import it too. The required tables and columns per axis are in
``REQUIRED`` below and in ``experiments/README.md``.

File layout::

    {"schema": 1, "axis": "axis-1-cache-capacity", "device": "mac",
     "created": "2026-11-07T15:27:00", "git_commit": "918204b",
     "system": {...}, "config": {...}, "prompts": {...}, "notes": "...",
     "tables": {"summary": {"columns": [...], "data": [[...], ...]}, ...}}
"""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path

import pandas as pd

SCHEMA = 1

# Device id -> label used in figures. The order is the plotting order (and colour order).
DEVICES = {
    "mac": "Mac (Apple Silicon, MLX)",
    "windows-igpu": "Windows iGPU (PyTorch)",
    "windows-dgpu": "Windows dGPU (PyTorch)",
}

# Tables (and the columns in them) that figures.py reads, per axis. A run may save
# more tables and more columns; these are the minimum every device must provide.
REQUIRED = {
    "axis-1-cache-capacity": {
        "summary": ["K", "hit_all", "hit_decode", "hit_prefill", "decode_tok_s", "ttft_mean_s",
                    "io_s", "wall_s", "io_share", "ms_per_load"],
        "per_request": ["K", "prompt", "hits", "misses", "hit_rate"],
        "working_set": ["tau", "W"],
    },
    "axis-2-eviction-policies": {
        "sweep": ["stream", "K", "policy", "hit_rate", "decode_hit_rate", "post_shift"],
        "real_runs": ["stream", "policy", "K", "hit_rate", "io_s", "decode_tok_s", "ttft_mean_s"],
    },
    "axis-3-prefill-chunking": {
        "runs": ["K", "C", "C_num", "prompt", "ttft_s", "GB_loaded", "scratch_GB",
                 "distinct_per_call", "chunk_footprint"],
    },
    "axis-4-io-compute-overlap": {
        "micro_reads": ["threads", "m", "ms"],
        "calls_by_m": ["mode", "phase", "m", "calls", "t_moe_ms"],
        "runs": ["mode", "decode_tok_s", "io_s"],
    },
    "axis-5-memory-tiering": {
        "runs": ["B", "strategy", "hit_rate", "GB_loaded", "decode_tok_s", "ttft_mean_s", "startup_s"],
        "replay": ["B", "strategy", "hit_rate", "GB_read"],
    },
    "axis-7-router-prefetching": {
        "prediction": ["phase", "predictor", "lookahead", "P", "layer", "n", "precision", "recall"],
        "runs": ["mode", "P", "lookahead", "threads", "decode_tok_s", "demand_load_s",
                 "prefetch_wait_s", "stall_s", "useful", "wasted"],
    },
    "axis-8-batch-contention": {
        "runs": ["K", "B", "agg_tok_s", "per_req_tok_s", "decode_hit_rate", "loads_per_step",
                 "reloads_per_step", "step_footprint", "step_p50_ms"],
    },
}


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, cwd=Path(__file__).parent, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _jsonable(value) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def config_from(namespace: dict) -> dict:
    """The run's configuration: every UPPER_CASE global that is plain JSON data.
    Call it right after the configuration cell, before the run adds other globals."""
    return {k: v for k, v in namespace.items()
            if k.isupper() and not k.startswith("_") and k not in ("CONFIG", "TABLES") and _jsonable(v)}


def load_prompts(axis_dir: Path | str) -> dict:
    """The axis's prompts (``<axis>/prompts.json``), shared by every device."""
    return json.loads((Path(axis_dir) / "prompts.json").read_text())


def check(axis: str, tables: dict) -> list[str]:
    """Missing required tables / columns for this axis (empty list = complete)."""
    problems = []
    for name, cols in REQUIRED.get(axis, {}).items():
        if name not in tables:
            problems.append(f"table '{name}' is missing")
            continue
        missing = [c for c in cols if c not in tables[name].columns]
        if missing:
            problems.append(f"table '{name}' lacks columns {missing}")
    return problems


def save(results_dir: Path | str, axis: str, device: str, tables: dict, *, config: dict | None = None,
         prompts: dict | None = None, system: dict | None = None, notes: str = "",
         strict: bool = True) -> Path:
    """Write ``<results_dir>/<device>.json``. ``tables`` maps a name to a DataFrame."""
    if device not in DEVICES:
        raise ValueError(f"device must be one of {list(DEVICES)}, got {device!r}")
    problems = check(axis, tables)
    if problems and strict:
        raise ValueError(f"{axis} results are incomplete: " + "; ".join(problems))
    doc = {
        "schema": SCHEMA, "axis": axis, "device": device,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(), "system": system or {}, "config": config or {},
        "prompts": prompts or {}, "notes": notes,
        # to_json turns NaN into null and numpy scalars into plain numbers
        "tables": {name: json.loads(pd.DataFrame(df).to_json(orient="split", index=False))
                   for name, df in tables.items()},
    }
    path = Path(results_dir) / f"{device}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1))
    return path


def load(results_dir: Path | str) -> dict:
    """{device: document} for every ``<device>.json`` in the folder, in DEVICES order.
    Each document's ``tables`` are DataFrames."""
    docs = {}
    for device in DEVICES:
        path = Path(results_dir) / f"{device}.json"
        if not path.exists():
            continue
        doc = json.loads(path.read_text())
        doc["tables"] = {name: pd.DataFrame(t["data"], columns=t["columns"])
                         for name, t in doc["tables"].items()}
        docs[device] = doc
    return docs


def table(results_dir: Path | str, name: str, docs: dict | None = None) -> pd.DataFrame:
    """One table from every device, stacked, with a ``device`` column first."""
    docs = load(results_dir) if docs is None else docs
    frames = [d["tables"][name].assign(device=dev) for dev, d in docs.items() if name in d["tables"]]
    if not frames:
        return pd.DataFrame(columns=["device"])
    out = pd.concat(frames, ignore_index=True)
    return out[["device"] + [c for c in out.columns if c != "device"]]
