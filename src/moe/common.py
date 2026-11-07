"""Framework-free core shared by every device (Mac / MLX and Windows / PyTorch).

Nothing here imports MLX or PyTorch, so the same eviction policies, trace replay and
workload helpers run on all three devices:

    storage index   parse_safetensors_header, TensorLoc, ExpertIndex
                    (byte range of every expert projection; headers only)
    policies        FIFO, LRU, LFU (optionally decayed), RSA (router-score-aware),
                    TieredPolicy (pinned static tier + dynamic tier), make_policy
    tracing         MoECall, Recorder, replay (exact), events_frame
    analysis        working_set_curve, knee_point, gini
    workloads       requests, chat_tokens, GenResult
    paths / io      repo_root, cache_dir, save_json, load_json

Backends build the cache on top of this: ``moe.mlx_cache`` (Mac). A PyTorch backend
for Windows should reuse these policies and ``replay`` unchanged.
"""

from __future__ import annotations

import json
import math
import struct
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from heapq import heappop, heappush
from pathlib import Path

import numpy as np

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

def repo_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (default: cwd) to the folder that holds ``models/``."""
    p = (start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "models").is_dir() and (cand / "experiments").is_dir():
            return cand
    raise FileNotFoundError("Could not find the repository root (models/ + experiments/).")


def cache_dir() -> Path:
    """Generated, git-ignored files (prompt pools, traces)."""
    d = repo_root() / "experiments" / ".cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Storage index: byte ranges of every expert (headers only, no tensor data)
# ─────────────────────────────────────────────────────────────────────────────

def parse_safetensors_header(path):
    """Read only the JSON header of a safetensors file (no tensor data)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return n, json.loads(f.read(n))


@dataclass(frozen=True)
class TensorLoc:
    file: str
    offset: int        # absolute byte offset of the tensor data in the file
    nbytes: int
    shape: tuple
    dtype: str


class ExpertIndex:
    """Where every expert projection lives on disk. Reads headers only."""

    def __init__(self, model_dir: Path | str):
        self.model_dir = Path(model_dir)
        self.shards = sorted(self.model_dir.glob("model*.safetensors"))
        if not self.shards:
            raise FileNotFoundError(f"No model*.safetensors in {self.model_dir}")
        self.loc: dict[tuple[int, int, str], TensorLoc] = {}
        self.non_expert_keys: dict[str, list[str]] = {}
        for shard in self.shards:
            hsize, meta = parse_safetensors_header(shard)
            base = 8 + hsize
            keys = []
            for name, info in meta.items():
                if name == "__metadata__":
                    continue
                if ".mlp.experts." in name and name.endswith(".weight"):
                    p = name.split(".")
                    s, e = info["data_offsets"]
                    self.loc[(int(p[2]), int(p[5]), p[6])] = TensorLoc(
                        str(shard), base + s, e - s, tuple(info["shape"]), info["dtype"])
                else:
                    keys.append(name)
            self.non_expert_keys[str(shard)] = keys

        layers = sorted({k[0] for k in self.loc})
        experts = sorted({k[1] for k in self.loc})
        self.num_layers, self.num_experts = len(layers), len(experts)
        self.total_experts = self.num_layers * self.num_experts
        g = self.loc[(0, 0, "gate_proj")]
        d = self.loc[(0, 0, "down_proj")]
        self.dtype_name = g.dtype          # 'F16' / 'BF16' / 'F32' (framework dtype is added per backend)
        self.intermediate, self.hidden = g.shape
        self.down_shape = d.shape
        self.expert_bytes = sum(self.loc[(0, 0, p)].nbytes for p in PROJECTIONS)
        assert len(self.loc) == 3 * self.total_experts, "missing expert tensors"

    @property
    def bank_bytes(self) -> int:
        return self.expert_bytes * self.total_experts

    def describe(self) -> str:
        return (f"{self.num_layers} layers x {self.num_experts} experts = {self.total_experts} "
                f"logical experts | {self.expert_bytes / 1e6:.2f} MB each ({self.dtype_name}) | "
                f"full bank {self.bank_bytes / 1e9:.1f} GB on disk")


# ─────────────────────────────────────────────────────────────────────────────
# Eviction policies (keys only). access(key, score, pinned) -> (hit, evicted)
# ``pinned`` keys are never chosen as victims (they are needed by the current
# MoE call). ``score`` is the router-gate mass of the key in this call; only the
# router-score policy uses it.
# ─────────────────────────────────────────────────────────────────────────────

class _Policy:
    name = "base"

    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self.hits = self.misses = self.evictions = 0

    def __len__(self):
        raise NotImplementedError

    def __contains__(self, key):
        raise NotImplementedError

    def access(self, key, score: float = 1.0, pinned=()):
        if key in self:
            self.hits += 1
            self._on_hit(key, score)
            return True, None
        self.misses += 1
        evicted = None
        if len(self) >= self.capacity:
            evicted = self._victim(pinned)
            self._remove(evicted)
            self.evictions += 1
        self._insert(key, score)
        return False, evicted

    def peek_victim(self, pinned=()):
        """Which key would be evicted next (without evicting)."""
        return self._victim(pinned, dry=True) if len(self) >= self.capacity else None


class FIFOPolicy(_Policy):
    name = "FIFO"

    def __init__(self, capacity):
        super().__init__(capacity)
        self._q = deque()
        self._set = set()

    def __len__(self):
        return len(self._set)

    def __contains__(self, key):
        return key in self._set

    def _on_hit(self, key, score):
        pass

    def _insert(self, key, score):
        self._q.append(key)
        self._set.add(key)

    def _remove(self, key):
        self._set.discard(key)
        if self._q[0] == key:
            self._q.popleft()
        else:                           # victim skipped a pinned head
            self._q.remove(key)

    def _victim(self, pinned, dry=False):
        for key in self._q:            # oldest first
            if key not in pinned:
                return key
        raise RuntimeError("all resident keys are pinned")


class LRUPolicy(_Policy):
    name = "LRU"

    def __init__(self, capacity):
        super().__init__(capacity)
        self._od = OrderedDict()

    def __len__(self):
        return len(self._od)

    def __contains__(self, key):
        return key in self._od

    def _on_hit(self, key, score):
        self._od.move_to_end(key)

    def _insert(self, key, score):
        self._od[key] = None

    def _remove(self, key):
        del self._od[key]

    def _victim(self, pinned, dry=False):
        for key in self._od:            # least recent first
            if key not in pinned:
                return key
        raise RuntimeError("all resident keys are pinned")


class _HeapPolicy(_Policy):
    """Min-heap of priorities with lazy invalidation."""

    def __init__(self, capacity):
        super().__init__(capacity)
        self._state = {}      # key -> policy state
        self._ver = {}
        self._next = 0
        self._heap = []
        self.t = 0            # logical time = number of accesses so far

    def __len__(self):
        return len(self._state)

    def __contains__(self, key):
        return key in self._state

    def access(self, key, score: float = 1.0, pinned=()):
        self.t += 1
        return super().access(key, score, pinned)

    def _push(self, key):
        self._ver[key] = self._next
        self._next += 1
        heappush(self._heap, (*self._priority(key), self._ver[key], key))

    def _remove(self, key):
        del self._state[key]
        self._ver.pop(key, None)

    def _victim(self, pinned, dry=False):
        held = []
        found = None
        while self._heap:
            item = heappop(self._heap)
            key, ver = item[-1], item[-2]
            if key not in self._state or self._ver.get(key) != ver:
                continue                  # stale entry
            held.append(item)
            if key not in pinned:
                found = key
                break
        for item in held:                  # put back everything we looked at
            heappush(self._heap, item)
        if found is None:
            raise RuntimeError("all resident keys are pinned")
        return found


class LFUPolicy(_HeapPolicy):
    """Least-frequently-used. With ``half_life`` (in accesses) counts decay
    exponentially: f(t) = sum 2^-((t - t_i)/H). Ties break toward older keys.
    Priorities are kept in log2 space so long streams cannot overflow."""

    def __init__(self, capacity, half_life: float | None = None):
        super().__init__(capacity)
        self.half_life = None if half_life is None else float(half_life)
        self.name = "LFU" if half_life is None else f"LFU-H{int(half_life)}"

    def _decayed(self, key):
        f, t_last = self._state[key]
        if self.half_life is None:
            return f
        return f * 2.0 ** (-(self.t - t_last) / self.half_life)

    def _priority(self, key):
        f, t_last = self._state[key]
        if self.half_life is None:
            return (f, t_last)
        # compare f*2^(-(t-t_last)/H) across keys at a common t  <=>  log2 f + t_last/H
        return (math.log2(f) + t_last / self.half_life, t_last)

    def _on_hit(self, key, score):
        self._state[key] = [self._decayed(key) + 1.0, self.t]
        self._push(key)

    def _insert(self, key, score):
        self._state[key] = [1.0, self.t]
        self._push(key)


class RouterScorePolicy(_HeapPolicy):
    """Router-score-aware eviction (new for Axis 2).

    Each resident expert keeps a decayed sum of the router-gate probability mass
    it has received:  s(t) = sum_i g_i * 2^-((t - t_i)/H).  The victim is the
    expert with the smallest s(t). Compared with decayed LFU, an access counts by
    how much the router actually relied on the expert (its gate weight summed
    over the tokens of the call) instead of 1, so experts the router picks only
    as a weak 7th/8th choice age out faster than experts it leans on.
    ``floor`` keeps the score positive for log-space priorities.
    """

    def __init__(self, capacity, half_life: float = 512.0, floor: float = 1e-6):
        super().__init__(capacity)
        self.half_life = float(half_life)
        self.floor = floor
        self.name = f"RSA-H{int(half_life)}"

    def _decayed(self, key):
        s, t_last = self._state[key]
        return s * 2.0 ** (-(self.t - t_last) / self.half_life)

    def _priority(self, key):
        s, t_last = self._state[key]
        return (math.log2(max(s, self.floor)) + t_last / self.half_life, t_last)

    def _on_hit(self, key, score):
        self._state[key] = [self._decayed(key) + float(score), self.t]
        self._push(key)

    def _insert(self, key, score):
        self._state[key] = [float(score), self.t]
        self._push(key)


class TieredPolicy(_Policy):
    """A static tier of permanently resident ("pinned") experts plus a dynamic
    tier managed by an inner policy. Pinned keys always hit and are never
    evicted; they must be loaded with ``ExpertCache.preload`` before the run."""

    def __init__(self, capacity, pinned_keys, inner: str = "LRU", **inner_kw):
        super().__init__(capacity)
        self.pinned = set(pinned_keys)
        dyn = self.capacity - len(self.pinned)
        if dyn < 1:
            raise ValueError("no room left for the dynamic tier")
        self.inner = make_policy(inner, dyn, **inner_kw)
        self.name = f"tiered({len(self.pinned)} pinned + {inner} {dyn})"

    def __len__(self):
        return len(self.pinned) + len(self.inner)

    def __contains__(self, key):
        return key in self.pinned or key in self.inner

    def access(self, key, score: float = 1.0, pinned=()):
        if key in self.pinned:
            self.hits += 1
            return True, None
        hit, ev = self.inner.access(key, score, pinned)
        if hit:
            self.hits += 1
        else:
            self.misses += 1
            if ev is not None:
                self.evictions += 1
        return hit, ev


POLICIES = {
    "FIFO": lambda c, **kw: FIFOPolicy(c),
    "LRU": lambda c, **kw: LRUPolicy(c),
    "LFU": lambda c, **kw: LFUPolicy(c, half_life=kw.get("half_life")),
    "RSA": lambda c, **kw: RouterScorePolicy(c, half_life=kw.get("half_life", 512.0)),
}


def make_policy(name: str, capacity: int, **kw) -> _Policy:
    """``name`` is FIFO / LRU / LFU / RSA; ``half_life`` applies to LFU and RSA."""
    return POLICIES[name](capacity, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Tracing: what each MoE layer call asked for
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MoECall:
    """What one MoE layer call asked for. Passed to observers."""
    layer: int
    n_tokens: int
    eids: list           # distinct experts, in fetch order
    gate_mass: list      # router probability mass per distinct expert (sum over tokens)
    indices: np.ndarray  # (n_tokens, top_k) expert ids
    scores: np.ndarray   # (n_tokens, top_k) router probabilities
    hits: int = 0
    misses: int = 0
    io_ms: float = 0.0
    groups: int = 1


# ─────────────────────────────────────────────────────────────────────────────
# Workloads and generation results
# ─────────────────────────────────────────────────────────────────────────────

def chat_tokens(tokenizer, text: str) -> list[int]:
    msgs = [{"role": "user", "content": text}]
    s = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return tokenizer.encode(s)


@dataclass
class GenResult:
    tokens: list
    text: str
    prompt_tokens: int
    ttft_s: float              # prefill + first token
    decode_s: float            # time for the remaining tokens
    step_ms: list = field(default_factory=list)   # wall time per decode step

    @property
    def decode_tok_s(self) -> float:
        n = len(self.step_ms)
        return n / self.decode_s if self.decode_s > 0 and n else float("nan")


def requests(prompts_by_domain: dict, n_per_domain: int | None = None, order: str = "interleave",
             offset: int = 0) -> list[dict]:
    """Turn {domain: [prompt, ...]} into a request list of {id, domain, text}.
    ``order='interleave'`` mixes domains round-robin (server-like traffic);
    ``order='blocks'`` serves each domain's requests back to back (domain shifts)."""
    lists = {d: ps[offset:offset + n_per_domain] if n_per_domain else ps[offset:]
             for d, ps in prompts_by_domain.items()}
    out = []
    if order == "blocks":
        for d, ps in lists.items():
            out += [{"domain": d, "text": t} for t in ps]
    else:
        for i in range(max(len(ps) for ps in lists.values())):
            for d, ps in lists.items():
                if i < len(ps):
                    out.append({"domain": d, "text": ps[i]})
    for i, r in enumerate(out):
        r["id"] = i
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Trace recording, exact replay and analysis
# ─────────────────────────────────────────────────────────────────────────────

class Recorder:
    """Observer that stores one compact record per MoE call."""

    def __init__(self):
        self.events: list[dict] = []
        self.tag: dict = {}

    def set(self, **tag):
        self.tag = dict(tag)

    def __call__(self, call: MoECall):
        self.events.append({
            **self.tag,
            "layer": call.layer, "n": call.n_tokens,
            "eids": call.eids, "mass": [round(m, 6) for m in call.gate_mass],
            "hits": call.hits, "misses": call.misses, "io_ms": call.io_ms,
        })


def replay(events, policy: _Policy):
    """Replay recorded calls through ``policy`` with the real cache's grouping and
    pinning rules. Returns a bool array of hits, one per (call, distinct expert)."""
    out = []
    K = policy.capacity
    for ev in events:
        eids, mass, layer = ev["eids"], ev["mass"], ev["layer"]
        for g0 in range(0, len(eids), K):
            grp = [(layer, e) for e in eids[g0:g0 + K]]
            pinned = set(grp)
            for key, m in zip(grp, mass[g0:g0 + K]):
                out.append(policy.access(key, m, pinned)[0])
    return np.asarray(out, dtype=bool)


def events_frame(events):
    """One row per (call, distinct expert) — the cache access stream."""
    import pandas as pd
    rows = []
    for c, ev in enumerate(events):
        for e, m in zip(ev["eids"], ev["mass"]):
            rows.append((c, ev.get("prompt", 0), ev.get("phase", ""), ev.get("step", 0),
                         ev["layer"], e, m, ev["n"]))
    return pd.DataFrame(rows, columns=["call", "prompt", "phase", "step", "layer", "eid", "mass", "n_tokens"])


def working_set_curve(events, windows=(1, 2, 4, 8, 16, 32, 64, 128), phase="decode"):
    """Denning working set W(tau): mean number of distinct (layer, expert) keys
    touched in a window of ``tau`` consecutive generated tokens (decode steps),
    within each prompt. Returns {tau: mean distinct keys}."""
    steps = []                          # list of key-sets, one per token step
    cur, cur_id = set(), None
    for ev in events:
        if ev.get("phase") != phase:
            continue
        sid = (ev.get("prompt"), ev.get("step"))
        if sid != cur_id and cur_id is not None:
            steps.append((cur_id[0], cur))
            cur = set()
        cur_id = sid
        cur.update((ev["layer"], e) for e in ev["eids"])
    if cur_id is not None:
        steps.append((cur_id[0], cur))
    out = {}
    for tau in windows:
        vals = []
        for s in range(0, len(steps) - tau + 1, max(1, tau // 2)):
            win = steps[s:s + tau]
            if win[0][0] != win[-1][0]:  # do not span two prompts
                continue
            vals.append(len(set().union(*(k for _, k in win))))
        out[tau] = float(np.mean(vals)) if vals else float("nan")
    return out


def knee_point(x, y, log_x: bool = True):
    """Kneedle-style knee: the point furthest above the chord joining the first
    and last points of an increasing, saturating curve (x on a log2 scale)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xs = np.log2(x) if log_x else x
    xn = (xs - xs[0]) / (xs[-1] - xs[0])
    yn = (y - y[0]) / ((y[-1] - y[0]) or 1.0)
    i = int(np.argmax(yn - xn))
    return x[i], i


def save_json(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def gini(values) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum())) - (n + 1) / n)
