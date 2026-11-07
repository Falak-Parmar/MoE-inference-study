"""Mac / MLX backend: the true K-slot expert cache used by every ``Mac.ipynb``.

It keeps the four concerns of the system separate:

    storage    ExpertIndex, ExpertReader   byte ranges of each expert inside the
                                           original safetensors shards; reads go
                                           straight from disk (page cache bypassed)
    residency  ExpertCache + policies      a global bank of K slots over the 1,024
                                           logical (layer, expert_id) experts
    routing    CachedMoE                   the model's own router (same math as
                                           mlx-lm), plus an observer hook
    compute    CachedMoE.switch            mx.gather_mm over the resident bank,
                                           indexed by slot

Everything framework-free (policies, replay, recorder, workloads, analysis) lives in
``moe.common`` and is re-exported here, so ``mc.make_policy``, ``mc.replay`` etc. work.

Design notes
------------
* K is GLOBAL: at most K logical experts are resident at any time, across all 16
  layers. The bank is preallocated as three MLX arrays of shape (K, ...), so the
  resident expert memory is exactly K x 12.6 MB (F16) and nothing else.
* A miss reads the expert's bytes from disk *directly into its slot* through a
  zero-copy NumPy view of the MLX buffer (Apple unified memory). There is no
  intermediate copy and no full-bank tensor anywhere.
* Because a slot is mutated in place, no pending MLX computation may still read
  the slot being overwritten. ``ExpertCache.ensure`` therefore evaluates the
  layer input (``sync``) before the first write. Every lazily-built graph that
  could read the bank is upstream of that input, *as long as the generation loop
  evaluates each step's token before building the next step* (all loops here do).
* Policies see keys only (no weights), so the same objects drive the real cache
  and the trace replay. ``replay`` reproduces the real cache's hit / miss /
  eviction counts exactly (the notebooks assert this).
"""

from __future__ import annotations

import fcntl
import gc
import os
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.tokenizer_utils import load as load_tokenizer
from mlx_lm.utils import _get_classes, load_config
from safetensors import safe_open

from . import common as _common
from .common import *  # noqa: F401,F403  (re-export the framework-free core)
from .common import _Policy

MODEL_NAME = "OLMoE-1B-7B-0125-Instruct"


def default_model_dir() -> Path:
    return repo_root() / "models" / MODEL_NAME


# ─────────────────────────────────────────────────────────────────────────────
# Storage: MLX dtypes + direct reads
# ─────────────────────────────────────────────────────────────────────────────

_DTYPES = {"F16": mx.float16, "BF16": mx.bfloat16, "F32": mx.float32}


class ExpertIndex(_common.ExpertIndex):
    """``moe.common.ExpertIndex`` plus the MLX dtype; defaults to the Mac checkpoint."""

    def __init__(self, model_dir: Path | str | None = None):
        super().__init__(model_dir or default_model_dir())
        self.dtype = _DTYPES[self.dtype_name]


class ExpertReader:
    """Reads expert bytes with ``pread``. ``direct=True`` sets F_NOCACHE so reads
    are not added to the macOS page cache (data that is *already* cached is still
    served from RAM; run ``sudo purge`` before a run for a truly cold disk)."""

    def __init__(self, index: ExpertIndex, direct: bool = True):
        self.index = index
        self.direct = direct
        self._fds: dict[str, int] = {}

    def _fd(self, path: str) -> int:
        fd = self._fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            if self.direct:
                fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
            self._fds[path] = fd
        return fd

    def read_into(self, layer: int, eid: int, dests) -> int:
        """Read the 3 projections of one expert into writable byte buffers."""
        total = 0
        for proj, dst in zip(PROJECTIONS, dests):
            loc = self.index.loc[(layer, eid, proj)]
            mv = memoryview(dst.reshape(-1).view(np.uint8))
            got = os.preadv(self._fd(loc.file), [mv], loc.offset)
            if got != loc.nbytes:
                raise IOError(f"short read for {(layer, eid, proj)}: {got}/{loc.nbytes}")
            total += got
        return total

    def read_arrays(self, layer: int, eid: int):
        """Fresh MLX arrays for one expert (used for reference checks only)."""
        out = []
        for proj in PROJECTIONS:
            loc = self.index.loc[(layer, eid, proj)]
            buf = np.empty(loc.nbytes, dtype=np.uint8)
            os.preadv(self._fd(loc.file), [memoryview(buf)], loc.offset)
            out.append(mx.array(buf.view(np.float16 if loc.dtype == "F16" else np.uint16)
                                .reshape(loc.shape)).view(_DTYPES[loc.dtype]))
        return out

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def purge_page_cache() -> bool:
    """Try to drop the OS file cache (needs passwordless sudo). Returns success."""
    try:
        r = subprocess.run(["sudo", "-n", "purge"], capture_output=True, timeout=120)
        return r.returncode == 0
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Residency: the global K-slot expert bank
# ─────────────────────────────────────────────────────────────────────────────

def max_safe_capacity(index: ExpertIndex, fraction: float = 0.70, reserve_gb: float = 1.5) -> int:
    """Largest K whose bank fits without swapping: the smaller of ``fraction`` of
    physical RAM and the RAM available right now (other apps count), minus a
    reserve for non-expert weights, KV cache and activations."""
    import psutil
    vm = psutil.virtual_memory()
    budget = min(vm.total * fraction, vm.available) - reserve_gb * 1e9
    return int(max(0, min(index.total_experts, budget // index.expert_bytes)))


class ExpertCache:
    """Global cache of at most ``capacity`` logical experts.

    The bank is three preallocated MLX arrays ``G, U, D`` of shape (K, ...).
    ``slot_of[(layer, eid)]`` says where a resident expert lives.
    """

    def __init__(self, index: ExpertIndex, capacity: int, policy: str | _Policy = "LRU",
                 reader: ExpertReader | None = None, io_threads: int = 1, **policy_kw):
        self.index = index
        self.capacity = int(capacity)
        if self.capacity < 1 or self.capacity > index.total_experts:
            raise ValueError(f"capacity must be in [1, {index.total_experts}]")
        self.policy = policy if isinstance(policy, _Policy) else make_policy(policy, capacity, **policy_kw)
        assert self.policy.capacity == self.capacity
        self.reader = reader or ExpertReader(index)
        self.io_threads = int(io_threads)
        self._pool = ThreadPoolExecutor(self.io_threads) if self.io_threads > 1 else None

        I, H = index.intermediate, index.hidden
        self.G = mx.zeros((self.capacity, I, H), dtype=index.dtype)
        self.U = mx.zeros((self.capacity, I, H), dtype=index.dtype)
        self.D = mx.zeros((self.capacity, *index.down_shape), dtype=index.dtype)
        mx.eval(self.G, self.U, self.D)
        # zero-copy writable views of the MLX buffers (unified memory)
        self._nG = np.array(self.G, copy=False)
        self._nU = np.array(self.U, copy=False)
        self._nD = np.array(self.D, copy=False)
        assert self._nG.flags.writeable

        self.slot_of: dict[tuple[int, int], int] = {}
        self.free = list(range(self.capacity - 1, -1, -1))
        self.reset_stats()

    # -- stats ---------------------------------------------------------------
    def reset_stats(self):
        self.hits = self.misses = self.evictions = 0
        self.bytes_loaded = 0
        self.io_ms = 0.0          # wall time spent inside disk reads
        self.load_ms: list[float] = []
        self.sync_ms = 0.0        # time waiting for pending compute before a write
        self.policy.hits = self.policy.misses = self.policy.evictions = 0

    def summary(self) -> dict:
        acc = self.hits + self.misses
        return {
            "K": self.capacity, "policy": self.policy.name,
            "accesses": acc, "hits": self.hits, "misses": self.misses,
            "hit_rate": self.hits / acc if acc else float("nan"),
            "evictions": self.evictions, "resident": len(self.slot_of),
            "bank_GB": self.bank_bytes / 1e9, "GB_loaded": self.bytes_loaded / 1e9,
            "io_ms": self.io_ms, "sync_ms": self.sync_ms,
            "ms_per_load": float(np.median(self.load_ms)) if self.load_ms else float("nan"),
        }

    @property
    def bank_bytes(self) -> int:
        return self.G.nbytes + self.U.nbytes + self.D.nbytes

    def __len__(self):
        return len(self.slot_of)

    def __contains__(self, key):
        return key in self.slot_of

    # -- decision + load -----------------------------------------------------
    def plan(self, keys, scores=None, protect=()):
        """Run the policy for one group of distinct keys (all pinned together).
        Returns (slots, misses) where misses is a list of (key, slot) to load.
        Evicted slots are freed immediately; the caller must load before compute.
        ``protect``: extra resident keys that must not be evicted (e.g. slots a
        background prefetch is still writing)."""
        if len(keys) > self.capacity:
            raise ValueError("group larger than capacity; split it first")
        pinned = set(keys) | set(protect)
        slots, misses = [], []
        for i, key in enumerate(keys):
            sc = 1.0 if scores is None else float(scores[i])
            hit, evicted = self.policy.access(key, sc, pinned)
            if evicted is not None:
                self.free.append(self.slot_of.pop(evicted))
                self.evictions += 1
            if hit:
                self.hits += 1
                slots.append(self.slot_of[key])
            else:
                self.misses += 1
                slot = self.free.pop()
                self.slot_of[key] = slot
                slots.append(slot)
                misses.append((key, slot))
        return slots, misses

    def _load_one(self, key, slot):
        t = time.perf_counter()
        n = self.reader.read_into(key[0], key[1], (self._nG[slot], self._nU[slot], self._nD[slot]))
        return n, (time.perf_counter() - t) * 1000

    def load(self, misses, sync=None):
        """Read missed experts into their slots. ``sync`` (an mx.array) is
        evaluated first so no pending computation still reads a reused slot."""
        if not misses:
            return 0.0
        if sync is not None:
            t = time.perf_counter()
            mx.eval(sync)
            self.sync_ms += (time.perf_counter() - t) * 1000
        t0 = time.perf_counter()
        if self._pool is None:
            results = [self._load_one(k, s) for k, s in misses]
        else:
            results = list(self._pool.map(lambda ks: self._load_one(*ks), misses))
        wall = (time.perf_counter() - t0) * 1000
        for n, ms in results:
            self.bytes_loaded += n
            self.load_ms.append(ms)
        self.io_ms += wall
        return wall

    def preload(self, keys) -> float:
        """Load ``keys`` into free slots without going through the policy (used
        for a static tier). Returns the wall time in ms; not counted as misses."""
        todo = []
        for key in keys:
            if key in self.slot_of:
                continue
            slot = self.free.pop()
            self.slot_of[key] = slot
            todo.append((key, slot))
        t0 = time.perf_counter()
        if self._pool is None:
            for k, s in todo:
                self._load_one(k, s)
        else:
            list(self._pool.map(lambda ks: self._load_one(*ks), todo))
        return (time.perf_counter() - t0) * 1000

    def ensure(self, keys, scores=None, sync=None, protect=()):
        """Make ``keys`` resident; returns their slots."""
        slots, misses = self.plan(keys, scores, protect)
        self.load(misses, sync)
        return slots

    def close(self):
        """Release the bank memory now (the model may still hold a reference to
        this object, so do not rely on garbage collection)."""
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
        self._nG = self._nU = self._nD = None
        self.G = self.U = self.D = None
        self.slot_of.clear()
        self.free = []
        gc.collect()
        mx.clear_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Routing + compute: drop-in replacement for OlmoeSparseMoeBlock
# ─────────────────────────────────────────────────────────────────────────────

def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


class CachedMoE(nn.Module):
    """Same routing math as mlx-lm's ``OlmoeSparseMoeBlock``; experts come from
    the global ``ExpertCache`` bank instead of a resident 64-expert tensor."""

    def __init__(self, gate: nn.Linear, layer: int, cache: ExpertCache | None,
                 top_k: int = 8, norm_topk_prob: bool = False):
        super().__init__()
        self.gate = gate
        self.layer = layer
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        object.__setattr__(self, "cache", cache)
        object.__setattr__(self, "observers", [])
        object.__setattr__(self, "route_wait_ms", 0.0)

    # router (identical to mlx-lm)
    def route(self, x_flat):
        logits = self.gate(x_flat)
        probs = mx.softmax(logits, axis=1, precise=True)
        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(-probs, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(probs, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        return inds, scores

    @staticmethod
    def switch(x_flat, slot_idx, G, U, D):
        """SwitchGLU over a slot bank (same op order as mlx-lm)."""
        x = mx.expand_dims(x_flat, (-2, -3))
        do_sort = slot_idx.size >= 64
        idx, inv = slot_idx, None
        if do_sort:
            x, idx, inv = _gather_sort(x, slot_idx)
        up = mx.gather_mm(x, U.swapaxes(-1, -2), rhs_indices=idx, sorted_indices=do_sort)
        gt = mx.gather_mm(x, G.swapaxes(-1, -2), rhs_indices=idx, sorted_indices=do_sort)
        y = mx.gather_mm(swiglu(gt, up), D.swapaxes(-1, -2), rhs_indices=idx, sorted_indices=do_sort)
        if do_sort:
            y = mx.unflatten(y[inv], 0, slot_idx.shape)
        return y.squeeze(-2)

    def __call__(self, x):
        B, L, Dm = x.shape
        x_flat = x.reshape(-1, Dm)
        inds, scores = self.route(x_flat)
        t = time.perf_counter()
        mx.eval(inds, scores)             # routing must be known on the CPU
        object.__setattr__(self, "route_wait_ms", self.route_wait_ms + (time.perf_counter() - t) * 1000)

        ind_np = np.array(inds)
        sc_np = np.array(scores.astype(mx.float32))
        eids, inv = np.unique(ind_np, return_inverse=True)
        mass = np.bincount(inv.reshape(-1), weights=sc_np.reshape(-1), minlength=len(eids))
        call = MoECall(self.layer, int(ind_np.shape[0]), eids.tolist(), mass.tolist(), ind_np, sc_np)

        cache = self.cache
        h0, m0, io0 = cache.hits, cache.misses, cache.io_ms
        n = len(eids)
        if n <= cache.capacity:
            keys = [(self.layer, int(e)) for e in eids]
            slots = cache.ensure(keys, mass, sync=x)
            lut = np.zeros(64, dtype=np.int32)
            lut[eids] = slots
            y = self.switch(x_flat, mx.array(lut[ind_np]), cache.G, cache.U, cache.D)
            y = (y * scores[..., None]).sum(axis=-2)
            call.groups = 1
        else:
            # More distinct experts than slots: process in groups of <= K.
            # Each group is computed and evaluated before the next group's loads
            # may overwrite its slots.
            y = mx.zeros_like(x_flat)
            ngroups = 0
            for g0 in range(0, n, cache.capacity):
                grp = eids[g0:g0 + cache.capacity]
                keys = [(self.layer, int(e)) for e in grp]
                slots = cache.ensure(keys, mass[g0:g0 + cache.capacity], sync=[x, y])
                lut = np.zeros(64, dtype=np.int32)
                in_grp = np.zeros(64, dtype=bool)
                lut[grp] = slots
                in_grp[grp] = True
                part = self.switch(x_flat, mx.array(lut[ind_np]), cache.G, cache.U, cache.D)
                w = scores * mx.array(in_grp[ind_np]).astype(scores.dtype)
                y = y + (part * w[..., None]).sum(axis=-2)
                mx.eval(y)
                ngroups += 1
            call.groups = ngroups

        call.hits = cache.hits - h0
        call.misses = cache.misses - m0
        call.io_ms = cache.io_ms - io0
        for obs in self.observers:
            obs(call)
        return y.reshape(B, L, Dm)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (non-expert weights only)
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_dir: Path | str | None = None, index: ExpertIndex | None = None):
    """Build OLMoE, load ONLY the non-expert weights (attention, norms, router,
    embeddings, head), and replace every MoE block with a ``CachedMoE``. The
    random expert tensors of the skeleton are never evaluated, so the expert bank
    is never materialised. Call ``attach_cache`` before running the model."""
    model_dir = Path(model_dir or default_model_dir())
    index = index or ExpertIndex(model_dir)
    config = load_config(model_dir)
    tokenizer = load_tokenizer(model_dir)
    model_cls, args_cls = _get_classes(config)
    args = args_cls.from_dict(config)
    model = model_cls(args)

    # Swap out the expert bank before anything is evaluated.
    for i, layer in enumerate(model.model.layers):
        gate = layer.mlp.gate
        layer.mlp = CachedMoE(gate, i, None, args.num_experts_per_tok, args.norm_topk_prob)
    gc.collect()

    for shard in index.shards:
        keys = index.non_expert_keys[str(shard)]
        with safe_open(shard, framework="numpy") as f:
            weights = [(k, mx.array(f.get_tensor(k))) for k in keys]
        model.load_weights(weights, strict=False)
        del weights
    mx.eval(model.parameters())
    gc.collect()
    mx.clear_cache()
    return model, tokenizer, index


def moe_blocks(model) -> list[CachedMoE]:
    return [layer.mlp for layer in model.model.layers]


def attach_cache(model, cache: ExpertCache, observers=()):
    """Point every layer at ``cache`` and set the observers."""
    for blk in moe_blocks(model):
        object.__setattr__(blk, "cache", cache)
        object.__setattr__(blk, "observers", list(observers))
        object.__setattr__(blk, "route_wait_ms", 0.0)


def detach_cache(model):
    """Drop the model's references to the current cache and observers."""
    for blk in moe_blocks(model):
        object.__setattr__(blk, "cache", None)
        object.__setattr__(blk, "observers", [])


def route_wait_ms(model) -> float:
    return sum(b.route_wait_ms for b in moe_blocks(model))


def non_expert_bytes(model) -> int:
    from mlx.utils import tree_flatten
    return sum(v.nbytes for _, v in tree_flatten(model.parameters()))


# ─────────────────────────────────────────────────────────────────────────────
# Generation (greedy, fixed length, explicit synchronisation every step)
# ─────────────────────────────────────────────────────────────────────────────
def generate(model, tokenizer, prompt_tokens, max_new_tokens: int = 32,
             prefill_chunk: int | None = None, on_step=None) -> GenResult:
    """Greedy decoding for exactly ``max_new_tokens`` (EOS is not a stop, so
    every run does the same amount of work). ``prefill_chunk`` splits the prompt
    into chunks of that many tokens (None = whole prompt in one pass).
    ``on_step(phase, i)`` is called before each forward pass."""
    toks = list(prompt_tokens)
    cache = make_prompt_cache(model)
    chunk = prefill_chunk or len(toks)

    t0 = time.perf_counter()
    logits = None
    for i, s in enumerate(range(0, len(toks), chunk)):
        if on_step:
            on_step("prefill", i)
        logits = model(mx.array(toks[s:s + chunk])[None], cache=cache)
        mx.eval(logits)
    tok = mx.argmax(logits[:, -1, :], axis=-1)
    out = [int(tok.item())]
    ttft = time.perf_counter() - t0

    step_ms = []
    t1 = time.perf_counter()
    for i in range(max_new_tokens - 1):
        if on_step:
            on_step("decode", i)
        ts = time.perf_counter()
        logits = model(mx.array([[out[-1]]]), cache=cache)
        tok = mx.argmax(logits[:, -1, :], axis=-1)
        out.append(int(tok.item()))      # .item() synchronises
        step_ms.append((time.perf_counter() - ts) * 1000)
    decode = time.perf_counter() - t1
    return GenResult(out, tokenizer.decode(out), len(toks), ttft, decode, step_ms)


def generate_batch(model, tokenizer, prompts_tokens, max_new_tokens: int = 32, on_step=None) -> dict:
    """Greedy decoding of B prompts *together* (one forward pass per step for the
    whole batch). Prompts are left-padded; mlx-lm's BatchKVCache masks the pads.
    Every step is synchronised. Returns tokens per request and timings."""
    from mlx_lm.models.cache import BatchKVCache
    B = len(prompts_tokens)
    lens = [len(t) for t in prompts_tokens]
    L = max(lens)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    left = [L - n for n in lens]
    x = mx.array([[pad] * lp + list(t) for lp, t in zip(left, prompts_tokens)])
    cache = [BatchKVCache(left) for _ in model.layers]

    t0 = time.perf_counter()
    if on_step:
        on_step("prefill", 0)
    logits = model(x, cache=cache)
    tok = mx.argmax(logits[:, -1, :], axis=-1)
    out = [[t] for t in tok.tolist()]
    ttft = time.perf_counter() - t0

    step_ms = []
    t1 = time.perf_counter()
    for i in range(max_new_tokens - 1):
        if on_step:
            on_step("decode", i)
        ts = time.perf_counter()
        logits = model(tok[:, None], cache=cache)
        tok = mx.argmax(logits[:, -1, :], axis=-1)
        for seq, t in zip(out, tok.tolist()):        # .tolist() synchronises
            seq.append(t)
        step_ms.append((time.perf_counter() - ts) * 1000)
    return {"tokens": out, "texts": [tokenizer.decode(o) for o in out], "ttft_s": ttft,
            "decode_s": time.perf_counter() - t1, "step_ms": step_ms, "left_pad": left}


def run_workload(model, tokenizer, cache: ExpertCache, prompts, max_new_tokens: int = 32,
                 prefill_chunk: int | None = None, observers=(), log=print):
    """Serve ``prompts`` one after another through ONE cache (state carries over
    between requests, as in a server). Returns (per-prompt DataFrame, Recorder)."""
    import pandas as pd
    rec = Recorder()
    attach_cache(model, cache, [rec, *observers])
    rows = []
    for i, p in enumerate(prompts):
        toks = chat_tokens(tokenizer, p["text"])
        h0, m0, io0, b0 = cache.hits, cache.misses, cache.io_ms, cache.bytes_loaded
        rw0 = route_wait_ms(model)

        def on_step(phase, j, _i=i, _d=p["domain"]):
            rec.set(prompt=_i, domain=_d, phase=phase, step=j)

        t = time.perf_counter()
        r = generate(model, tokenizer, toks, max_new_tokens, prefill_chunk, on_step=on_step)
        wall = time.perf_counter() - t
        hits, misses = cache.hits - h0, cache.misses - m0
        io_s = (cache.io_ms - io0) / 1000
        rows.append({
            "prompt": i, "domain": p["domain"], "prompt_tokens": r.prompt_tokens,
            "new_tokens": len(r.tokens), "ttft_s": r.ttft_s, "decode_s": r.decode_s,
            "decode_tok_s": r.decode_tok_s, "wall_s": wall,
            "hits": hits, "misses": misses, "hit_rate": hits / max(1, hits + misses),
            "GB_loaded": (cache.bytes_loaded - b0) / 1e9, "io_s": io_s,
            "route_wait_s": (route_wait_ms(model) - rw0) / 1000,
            "non_io_s": wall - io_s, "text": r.text,
        })
        if log:
            log(f"  [{i + 1}/{len(prompts)}] {p['domain']:13s} {r.prompt_tokens:5d} tok | "
                f"TTFT {r.ttft_s:6.2f}s | {r.decode_tok_s:5.2f} tok/s | hit {rows[-1]['hit_rate']:6.1%} | "
                f"io {io_s:6.2f}s of {wall:6.2f}s")
    return pd.DataFrame(rows), rec


# ─────────────────────────────────────────────────────────────────────────────
# Measurement helpers
# ─────────────────────────────────────────────────────────────────────────────

def memory_snapshot() -> dict:
    import psutil
    return {
        "mlx_active_GB": mx.get_active_memory() / 1e9,
        "mlx_peak_GB": mx.get_peak_memory() / 1e9,
        "mlx_cache_GB": mx.get_cache_memory() / 1e9,
        "process_rss_GB": psutil.Process().memory_info().rss / 1e9,
    }


def system_info() -> dict:
    import psutil
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        chip = platform.processor()
    return {
        "chip": chip, "ram_GB": round(psutil.virtual_memory().total / 1e9, 1),
        "macOS": platform.mac_ver()[0], "python": platform.python_version(),
        "mlx": mx.__version__,
    }


def free_memory():
    gc.collect()
    mx.clear_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers (used at the top of every notebook)
# ─────────────────────────────────────────────────────────────────────────────

def reference_layer_output(index: ExpertIndex, blk: CachedMoE, x, reader: ExpertReader | None = None):
    """The same MoE layer computed the stock mlx-lm way: all 64 experts of the
    layer stacked into one resident tensor (only ONE layer, ~0.8 GB, for checking)."""
    from mlx_lm.models.switch_layers import SwitchGLU
    reader = reader or ExpertReader(index, direct=True)     # do not leave experts in the page cache
    E = index.num_experts
    sw = SwitchGLU(index.hidden, index.intermediate, E)
    ws = [reader.read_arrays(blk.layer, e) for e in range(E)]
    sw.gate_proj.weight = mx.stack([w[0] for w in ws])
    sw.up_proj.weight = mx.stack([w[1] for w in ws])
    sw.down_proj.weight = mx.stack([w[2] for w in ws])
    x_flat = x.reshape(-1, x.shape[-1])
    inds, scores = blk.route(x_flat)
    y = sw(x_flat, inds)
    y = (y * scores[..., None]).sum(axis=-2)
    out = y.reshape(x.shape)
    mx.eval(out)
    del sw, ws
    return out


def check_layer_exactness(model, index: ExpertIndex, capacity: int = 64, n_tokens: int = 37,
                          layer: int = 3, seed: int = 0) -> float:
    """Max |cached - reference| for one layer on random hidden states (should be 0)."""
    blk = moe_blocks(model)[layer]
    mx.random.seed(seed)
    x = (mx.random.normal((1, n_tokens, index.hidden)) * 0.5).astype(index.dtype)
    ref = reference_layer_output(index, blk, x)
    cache = ExpertCache(index, capacity, "LRU")
    old = blk.cache
    object.__setattr__(blk, "cache", cache)
    try:
        got = blk(x)
        mx.eval(got)
    finally:
        object.__setattr__(blk, "cache", old)
    diff = float(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max().item())
    cache.close()
    free_memory()
    return diff
