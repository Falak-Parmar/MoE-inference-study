"""Windows / PyTorch backend: the true K-slot expert cache used by every ``Windows_DGPU.ipynb``.

It mirrors ``moe.mlx_cache`` (same names, same arguments), so a Windows notebook is the
Mac notebook with ``mc`` replaced by ``tc``. The four concerns stay separate:

    storage    ExpertIndex, ExpertReader   byte ranges of each expert inside the
                                           original safetensors shards; reads go
                                           straight from disk (OS file cache bypassed)
    residency  ExpertCache + policies      a global bank of K slots in VRAM over the
                                           1,024 logical (layer, expert_id) experts;
               HostTier                    optional bounded RAM tier (pinned memory)
    routing    CachedMoE                   the model's own router (same math as
                                           transformers' OlmoeSparseMoeBlock)
    compute    CachedMoE.expert_out        transformers' eager per-expert loop over the
                                           VRAM bank, indexed by slot

Everything framework-free (policies, replay, recorder, workloads, analysis) lives in
``moe.common`` and is re-exported here, so ``tc.make_policy``, ``tc.replay`` etc. work.

Design notes
------------
* K is GLOBAL and counts experts in the fastest tier (VRAM on the dGPU). The bank is
  two preallocated tensors, ``GU`` (K, 2I, H) = [gate; up] and ``D`` (K, H, I), so the
  resident expert memory is exactly K x 12.6 MB (BF16) and nothing else. The full bank
  is never loaded, neither in VRAM nor in RAM.
* Data path of a miss: disk -> pinned staging buffer (one per reader thread) -> async
  copy on that thread's CUDA stream into the slot. Reads are unbuffered
  (``FILE_FLAG_NO_BUFFERING`` on Windows, ``O_DIRECT`` on Linux, ``F_NOCACHE`` on macOS),
  so the OS file cache never quietly holds the bank. Read and copy times are recorded
  separately (``disk_s`` and ``h2d_s``).
* Slot reuse is safe without extra events: a layer only loads after its routing has
  been copied to the CPU, which waits for every kernel queued before it on the compute
  stream. A layer call with more distinct experts than K processes them in groups and
  synchronises between groups.
* Compute follows transformers' eager expert loop op for op (same expert order, same
  gather, fused or split gate/up as the installed version does it, same index_add_), so
  one cached layer is bit-identical to the stock layer (checked at the top of every
  notebook).
* Policies see keys only, so the same objects drive the real cache and ``replay``,
  which reproduces the real cache's hit / miss / eviction counts exactly.
"""

from __future__ import annotations

import copy
import gc
import os
import platform
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import common as _common
from .common import *  # noqa: F401,F403  (re-export the framework-free core)
from .common import _Policy

MODEL_NAME = "OLMoE-1B-7B-0125-Instruct"
HF_REPO = "allenai/OLMoE-1B-7B-0125-Instruct"
ALIGN = 4096            # sector alignment for unbuffered reads (covers 512e and 4Kn drives)


def default_model_dir() -> Path:
    return repo_root() / "models" / MODEL_NAME


def download_model(model_dir: Path | str | None = None) -> Path:
    """Fetch the BF16 checkpoint from Hugging Face if it is not already there.
    README.md is not downloaded (the repository keeps its own)."""
    model_dir = Path(model_dir or default_model_dir())
    have = (model_dir / "config.json").exists() and (model_dir / "tokenizer.json").exists() \
        and list(model_dir.glob("model*.safetensors"))
    if not have:
        from huggingface_hub import snapshot_download
        snapshot_download(HF_REPO, local_dir=model_dir, allow_patterns=["*.json", "*.safetensors"])
    return model_dir


# ─────────────────────────────────────────────────────────────────────────────
# Device helpers
# ─────────────────────────────────────────────────────────────────────────────

def default_device() -> torch.device:
    """CUDA if present (the dGPU), else MPS / CPU (only for smoke tests elsewhere).
    The environment variable ``MOE_DEVICE`` (e.g. ``cpu``) overrides the choice."""
    if os.environ.get("MOE_DEVICE"):
        return torch.device(os.environ["MOE_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device=None):
    """Wait for every queued kernel (call before reading a timer)."""
    d = torch.device(device) if device is not None else default_device()
    if d.type == "cuda":
        torch.cuda.synchronize(d)
    elif d.type == "mps":
        torch.mps.synchronize()


def model_device(model) -> torch.device:
    return next(model.parameters()).device


# ─────────────────────────────────────────────────────────────────────────────
# Storage: torch dtypes + unbuffered reads
# ─────────────────────────────────────────────────────────────────────────────

_DTYPES = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}


class ExpertIndex(_common.ExpertIndex):
    """``moe.common.ExpertIndex`` plus the torch dtype; defaults to the Windows checkpoint."""

    def __init__(self, model_dir: Path | str | None = None):
        super().__init__(model_dir or default_model_dir())
        self.dtype = _DTYPES[self.dtype_name]


class HostBuffer:
    """A CPU byte buffer whose start is ALIGN-aligned; pinned (page-locked) when
    ``pin`` so that copies to the GPU are truly asynchronous."""

    def __init__(self, nbytes: int, pin: bool = False, register: bool = False):
        """``register=True`` pins an ordinary allocation with ``cudaHostRegister`` (exact
        size; for large blocks), instead of PyTorch's pinned allocator, which rounds every
        allocation up to a power of two."""
        self.nbytes = int(nbytes)
        self.registered = False
        raw = torch.empty(self.nbytes + ALIGN, dtype=torch.uint8, pin_memory=pin and not register)
        if pin and register:
            try:
                torch.cuda.check_error(torch.cuda.cudart().cudaHostRegister(raw.data_ptr(), raw.numel(), 0))
                self.registered = True
            except Exception as e:
                print(f"[HostBuffer] cudaHostRegister failed ({e}); this block stays pageable")
        shift = (-raw.data_ptr()) % ALIGN
        self._raw = raw
        self.t = raw[shift:shift + self.nbytes]          # torch view (for copies)
        self.np = self.t.numpy()                          # numpy view (for reads)
        assert self.t.data_ptr() % ALIGN == 0

    def slice(self, offset: int, nbytes: int) -> "HostBuffer":
        """An aligned sub-buffer sharing this memory (``offset`` must be a multiple of ALIGN)."""
        assert offset % ALIGN == 0 and offset + nbytes <= self.nbytes
        sub = object.__new__(HostBuffer)
        sub.nbytes, sub.registered, sub._raw = nbytes, False, None
        sub.t = self.t[offset:offset + nbytes]
        sub.np = self.np[offset:offset + nbytes]
        return sub

    def release(self):
        if self.registered:
            torch.cuda.cudart().cudaHostUnregister(self._raw.data_ptr())
            self.registered = False
        self._raw = self.t = self.np = None


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _OVERLAPPED(ctypes.Structure):
        _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
                    ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD),
                    ("hEvent", wintypes.HANDLE)]

    _k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                 wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    _k32.CreateFileW.restype = wintypes.HANDLE
    _k32.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                              ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(_OVERLAPPED)]
    _k32.ReadFile.restype = wintypes.BOOL
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.CloseHandle.restype = wintypes.BOOL
    _INVALID_HANDLE = wintypes.HANDLE(-1).value
    _GENERIC_READ, _FILE_SHARE_READ, _OPEN_EXISTING = 0x80000000, 0x1, 3
    _FILE_FLAG_NO_BUFFERING, _FILE_ATTRIBUTE_NORMAL = 0x20000000, 0x80
    _ERROR_HANDLE_EOF = 38


class ExpertReader:
    """Reads the three projections of one expert into a ``HostBuffer``.

    ``direct=True`` bypasses the OS file cache:
      * Windows: ``CreateFileW(FILE_FLAG_NO_BUFFERING)`` + ``ReadFile`` at an offset
      * Linux:   ``O_DIRECT`` + ``preadv``
      * macOS:   ``F_NOCACHE`` + ``preadv`` (used only to smoke-test the code on a Mac)
    Unbuffered reads need sector-aligned offsets, lengths and buffers, so each tensor's
    byte range is widened to ALIGN and read into an aligned region of the buffer; the
    tensor then starts a few bytes into that region. If the unbuffered open fails the
    reader falls back to ordinary buffered reads and says so in ``mode``.
    File handles are per thread, so several reader threads can read at once.
    """

    def __init__(self, index: ExpertIndex, direct: bool = True):
        self.index = index
        self.direct = bool(direct)
        self.mode = self._default_mode() if self.direct else "buffered"
        self.align = ALIGN if self.mode != "buffered" else 1
        # fixed buffer layout: region i holds projection i (widened to ALIGN)
        self._bounds = []
        for proj in PROJECTIONS:
            n = index.loc[(0, 0, proj)].nbytes
            self._bounds.append(-(-n // ALIGN) * ALIGN + ALIGN)
        self.buf_bytes = sum(self._bounds)
        self._tls = threading.local()
        self._handles = []
        self._lock = threading.Lock()

    @staticmethod
    def _default_mode() -> str:
        if sys.platform == "win32":
            return "windows-unbuffered"
        if sys.platform.startswith("linux"):
            return "linux-o_direct"
        if sys.platform == "darwin":
            return "macos-nocache"
        return "buffered"

    # -- file handles (one per thread and file) ------------------------------
    def _open(self, path: str):
        if self.mode == "windows-unbuffered":
            h = _k32.CreateFileW(path, _GENERIC_READ, _FILE_SHARE_READ, None, _OPEN_EXISTING,
                                 _FILE_ATTRIBUTE_NORMAL | _FILE_FLAG_NO_BUFFERING, None)
            if h is None or h == _INVALID_HANDLE:
                raise OSError(ctypes.get_last_error(), f"CreateFileW failed for {path}")
            return ("win", h)
        if self.mode == "linux-o_direct":
            return ("fd", os.open(path, os.O_RDONLY | os.O_DIRECT))
        if self.mode == "macos-nocache":
            import fcntl
            fd = os.open(path, os.O_RDONLY)
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
            return ("fd", fd)
        if hasattr(os, "preadv"):
            return ("fd", os.open(path, os.O_RDONLY))
        return ("file", open(path, "rb", buffering=0))

    def _handle(self, path: str):
        hs = getattr(self._tls, "handles", None)
        if hs is None:
            hs = self._tls.handles = {}
        h = hs.get(path)
        if h is None:
            try:
                h = self._open(path)
            except OSError as e:
                if self.mode == "buffered":
                    raise
                print(f"[ExpertReader] unbuffered open failed ({e}); falling back to buffered reads")
                self.mode, self.align = "buffered", 1
                h = self._open(path)
            hs[path] = h
            with self._lock:
                self._handles.append(h)
        return h

    def _pread(self, path: str, buf: np.ndarray, pos: int, start: int, length: int) -> int:
        """Read ``length`` bytes at file offset ``start`` into ``buf[pos:]``. Returns bytes read
        (fewer than ``length`` only at end of file)."""
        kind, h = self._handle(path)
        got = 0
        while got < length:
            if kind == "win":
                n = wintypes.DWORD(0)
                ov = _OVERLAPPED()
                off = start + got
                ov.Offset, ov.OffsetHigh = off & 0xFFFFFFFF, off >> 32
                ok = _k32.ReadFile(h, ctypes.c_void_p(buf.ctypes.data + pos + got), length - got,
                                   ctypes.byref(n), ctypes.byref(ov))
                if not ok:
                    err = ctypes.get_last_error()
                    if err == _ERROR_HANDLE_EOF:
                        break
                    raise OSError(err, f"ReadFile failed at offset {off} of {path}")
                k = n.value
            elif kind == "fd":
                k = os.preadv(h, [memoryview(buf[pos + got:pos + length])], start + got)
            else:
                h.seek(start + got)
                k = h.readinto(memoryview(buf[pos + got:pos + length]))
            got += k
            if k < length - (got - k):
                break       # short read = end of file (retrying would be unaligned for unbuffered I/O)
        return got

    def read_into(self, layer: int, eid: int, buf: np.ndarray):
        """Read one expert into ``buf`` (a ``HostBuffer.np``). Returns (layout, nbytes):
        ``layout[proj] = (byte offset of the tensor in buf, nbytes)``."""
        layout, total, pos = {}, 0, 0
        A = self.align
        for proj, bound in zip(PROJECTIONS, self._bounds):
            loc = self.index.loc[(layer, eid, proj)]
            a = loc.offset // A * A
            e = -(-(loc.offset + loc.nbytes) // A) * A
            got = self._pread(loc.file, buf, pos, a, e - a)
            if got < loc.offset + loc.nbytes - a:
                raise IOError(f"short read for {(layer, eid, proj)}: {got} < {loc.offset + loc.nbytes - a}")
            layout[proj] = (pos + loc.offset - a, loc.nbytes)
            total += loc.nbytes
            pos += bound
        return layout, total

    def read_tensors(self, layer: int, eid: int):
        """Fresh CPU tensors (gate, up, down) for one expert (reference checks only)."""
        buf = HostBuffer(self.buf_bytes)
        layout, _ = self.read_into(layer, eid, buf.np)
        return [_view(buf, layout[p], self.index.loc[(layer, eid, p)].shape, self.index.dtype).clone()
                for p in PROJECTIONS]

    def close(self):
        with self._lock:
            for kind, h in self._handles:
                try:
                    if kind == "win":
                        _k32.CloseHandle(h)
                    elif kind == "fd":
                        os.close(h)
                    else:
                        h.close()
                except Exception:
                    pass
            self._handles.clear()
        self._tls = threading.local()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _view(buf: HostBuffer, entry, shape, dtype):
    pos, n = entry
    t = buf.t[pos:pos + n]
    if t.storage_offset() % torch.empty((), dtype=dtype).element_size():
        t = t.clone()     # tensor starts at an odd byte (unpadded safetensors header); HF files are aligned
    return t.view(dtype).view(shape)


_NON_CUDA_COPY_LOCK = threading.Lock()


def purge_page_cache() -> bool:
    """Nothing to purge: every expert read is unbuffered (see ``ExpertReader``), so the
    OS file cache never holds expert bytes. Returns False to say no purge ran."""
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Residency: the global K-slot bank in VRAM, plus an optional RAM tier
# ─────────────────────────────────────────────────────────────────────────────

def max_safe_capacity(index: ExpertIndex, device=None, fraction: float = 0.95,
                      reserve_gb: float = 0.6) -> int:
    """Largest K whose bank fits in the fast tier right now. On CUDA: the smaller of
    ``fraction`` of VRAM and the VRAM free now, minus a reserve for KV cache and
    activations (call after ``load_model``, before any bank exists). Elsewhere: RAM,
    as on the Mac."""
    d = torch.device(device) if device is not None else default_device()
    if d.type == "cuda":
        free, total = torch.cuda.mem_get_info(d)
        budget = min(total * fraction, free) - reserve_gb * 1e9
    else:
        import psutil
        vm = psutil.virtual_memory()
        budget = min(vm.total * 0.70, vm.available) - 1.5e9
    return int(max(0, min(index.total_experts, budget // index.expert_bytes)))


def max_host_capacity(index: ExpertIndex, fraction: float = 0.5, reserve_gb: float = 4.0) -> int:
    """Largest RAM tier (experts in pinned RAM) that leaves room for everything else."""
    import psutil
    vm = psutil.virtual_memory()
    budget = min(vm.total * fraction, vm.available - reserve_gb * 1e9)
    return int(max(0, min(index.total_experts - 1, budget // index.expert_bytes)))


class HostTier:
    """A bounded RAM tier between VRAM and disk: ``capacity`` experts in pinned host
    buffers, managed by its own policy (LRU). A VRAM miss that hits here costs only a
    host-to-device copy; a miss here reads from disk into the tier first. It never
    holds the full bank (capacity < 1,024)."""

    def __init__(self, index: ExpertIndex, capacity: int, policy: str = "LRU",
                 reader: ExpertReader | None = None, pin: bool | None = None):
        self.index = index
        self.capacity = int(capacity)
        if not 1 <= self.capacity < index.total_experts:
            raise ValueError(f"RAM tier capacity must be in [1, {index.total_experts - 1}]")
        self.policy_name = policy
        self.policy = make_policy(policy, self.capacity)
        pin = torch.cuda.is_available() if pin is None else pin
        nbytes = (reader or ExpertReader(index)).buf_bytes          # a multiple of ALIGN
        t = time.perf_counter()
        self.block = HostBuffer(nbytes * self.capacity, pin, register=True)   # one exact-size pinned block
        self.bufs = [self.block.slice(i * nbytes, nbytes) for i in range(self.capacity)]
        self.alloc_s = time.perf_counter() - t
        self.pinned = self.block.registered
        self.layout: dict[int, dict] = {}
        self.slot_of: dict[tuple[int, int], int] = {}
        self.free = list(range(self.capacity - 1, -1, -1))
        self.hits = self.misses = 0

    def access(self, key, pinned=()):
        """Decide where ``key`` comes from. Returns (hit, host slot)."""
        hit, ev = self.policy.access(key, 1.0, pinned)
        if ev is not None:
            s = self.slot_of.pop(ev)
            self.layout.pop(s, None)
            self.free.append(s)
        if hit:
            self.hits += 1
            return True, self.slot_of[key]
        self.misses += 1
        s = self.free.pop()
        self.slot_of[key] = s
        return False, s

    def reset(self):
        """Empty the tier (keeps the buffers), so each run starts cold."""
        self.policy = make_policy(self.policy_name, self.capacity)
        self.layout, self.slot_of = {}, {}
        self.free = list(range(self.capacity - 1, -1, -1))
        self.hits = self.misses = 0

    @property
    def bytes(self) -> int:
        return self.capacity * self.index.expert_bytes

    def close(self):
        self.bufs, self.layout, self.slot_of, self.free = [], {}, {}, []
        self.block.release()
        gc.collect()


class ExpertCache:
    """Global cache of at most ``capacity`` logical experts in VRAM.

    The bank is two preallocated tensors ``GU`` (K, 2I, H) and ``D`` (K, H, I).
    ``slot_of[(layer, eid)]`` says where a resident expert lives. With ``host_tier``
    a VRAM miss is served from the RAM tier when it can be, else from disk.
    """

    def __init__(self, index: ExpertIndex, capacity: int, policy: str | _Policy = "LRU",
                 reader: ExpertReader | None = None, io_threads: int = 1, device=None,
                 host_tier: HostTier | None = None, **policy_kw):
        self.index = index
        self.capacity = int(capacity)
        if self.capacity < 1 or self.capacity > index.total_experts:
            raise ValueError(f"capacity must be in [1, {index.total_experts}]")
        self.policy = policy if isinstance(policy, _Policy) else make_policy(policy, capacity, **policy_kw)
        assert self.policy.capacity == self.capacity
        self.reader = reader or ExpertReader(index)
        self.device = torch.device(device) if device is not None else default_device()
        self.io_threads = int(io_threads)
        self.host_tier = host_tier

        I, H = index.intermediate, index.hidden
        self.GU = torch.empty((self.capacity, 2 * I, H), dtype=index.dtype, device=self.device)
        self.D = torch.empty((self.capacity, *index.down_shape), dtype=index.dtype, device=self.device)
        self._tls = threading.local()      # per-thread pinned staging buffer + copy stream
        self._stages = []
        self._src: dict = {}               # key -> (host slot, host hit) decided in plan()

        self.slot_of: dict[tuple[int, int], int] = {}
        self.free = list(range(self.capacity - 1, -1, -1))
        self._stage()                      # allocate staging now, not inside a timed load
        self._pool = start_pool(self, self.io_threads) if self.io_threads > 1 else None
        self.reset_stats()

    # -- stats ---------------------------------------------------------------
    def reset_stats(self):
        self.hits = self.misses = self.evictions = 0
        self.bytes_loaded = 0
        self.io_ms = 0.0          # wall time spent loading misses (disk read + copy to VRAM)
        self.load_ms: list[float] = []
        self._detail: list[tuple] = []     # per load: (disk bytes, disk ms, h2d ms)
        self.sync_ms = 0.0
        self.policy.hits = self.policy.misses = self.policy.evictions = 0
        if self.host_tier is not None:
            self.host_tier.hits = self.host_tier.misses = 0

    def summary(self) -> dict:
        acc = self.hits + self.misses
        det = np.array(self._detail, dtype=float).reshape(-1, 3)
        out = {
            "K": self.capacity, "policy": self.policy.name,
            "accesses": acc, "hits": self.hits, "misses": self.misses,
            "hit_rate": self.hits / acc if acc else float("nan"),
            "evictions": self.evictions, "resident": len(self.slot_of),
            "bank_GB": self.bank_bytes / 1e9, "GB_loaded": self.bytes_loaded / 1e9,
            "io_ms": self.io_ms, "sync_ms": self.sync_ms,
            "ms_per_load": float(np.median(self.load_ms)) if self.load_ms else float("nan"),
            "disk_GB": float(det[:, 0].sum()) / 1e9, "disk_s": float(det[:, 1].sum()) / 1000,
            "h2d_s": float(det[:, 2].sum()) / 1000,
            "ms_per_disk_read": float(np.median(det[det[:, 0] > 0, 1])) if (det[:, 0] > 0).any() else float("nan"),
            "ms_per_h2d": float(np.median(det[:, 2])) if len(det) else float("nan"),
        }
        if self.host_tier is not None:
            h = self.host_tier
            out.update({"ram_tier": h.capacity, "ram_hits": h.hits, "ram_misses": h.misses,
                        "ram_hit_rate": h.hits / max(1, h.hits + h.misses)})
        return out

    @property
    def bank_bytes(self) -> int:
        return self.GU.numel() * self.GU.element_size() + self.D.numel() * self.D.element_size()

    def __len__(self):
        return len(self.slot_of)

    def __contains__(self, key):
        return key in self.slot_of

    # -- decision + load -----------------------------------------------------
    def plan(self, keys, scores=None, protect=()):
        """Run the policy for one group of distinct keys (all pinned together).
        Returns (slots, misses) where misses is a list of (key, slot) to load.
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
                if self.host_tier is not None:
                    hhit, hslot = self.host_tier.access(key, pinned)
                    self._src[key] = (hslot, hhit)
        return slots, misses

    def _stage(self):
        st = getattr(self._tls, "st", None)
        if st is None:
            cuda = self.device.type == "cuda"
            st = (HostBuffer(self.reader.buf_bytes, pin=cuda),
                  torch.cuda.Stream(self.device) if cuda else None)
            self._tls.st = st
            self._stages.append(st)
        return st

    def _copy_to_slot(self, buf: HostBuffer, layout, key, slot, stream):
        I = self.index.intermediate
        loc = self.index.loc
        src = [_view(buf, layout[p], loc[(key[0], key[1], p)].shape, self.index.dtype) for p in PROJECTIONS]
        dst = [self.GU[slot, :I], self.GU[slot, I:], self.D[slot]]
        if stream is not None:
            with torch.cuda.stream(stream):
                for d, s in zip(dst, src):
                    d.copy_(s, non_blocking=True)
            stream.synchronize()          # the slot is ready and the staging buffer is free again
        else:
            # MPS / CPU (smoke tests off Windows): Metal aborts if two threads drive the GPU
            # at once, so these copies are serialised. The CUDA path above stays parallel.
            with _NON_CUDA_COPY_LOCK:
                for d, s in zip(dst, src):
                    d.copy_(s)
                synchronize(self.device)

    def _load_one(self, key, slot):
        """Disk (or RAM tier) -> VRAM slot. Returns (bytes, ms). Thread-safe."""
        t0 = time.perf_counter()
        stage, stream = self._stage()
        disk = 0
        src = self._src.pop(key, None) if self.host_tier is not None else None
        if src is not None:
            hslot, hhit = src
            buf = self.host_tier.bufs[hslot]
            if hhit:
                layout = self.host_tier.layout[hslot]
            else:
                layout, disk = self.reader.read_into(key[0], key[1], buf.np)
                self.host_tier.layout[hslot] = layout
        else:
            buf = stage
            layout, disk = self.reader.read_into(key[0], key[1], buf.np)
        t1 = time.perf_counter()
        self._copy_to_slot(buf, layout, key, slot, stream)
        t2 = time.perf_counter()
        self._detail.append((disk, (t1 - t0) * 1000, (t2 - t1) * 1000))
        return self.index.expert_bytes, (t2 - t0) * 1000

    def load(self, misses, sync=None):
        """Load missed experts into their slots (``sync`` is accepted for API parity with
        the Mac; routing has already synchronised the compute stream)."""
        if not misses:
            return 0.0
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
        """Load ``keys`` into free slots without going through the policy (used for a
        static tier). Straight from disk. Returns the wall time in ms; not counted as misses."""
        todo = []
        for key in keys:
            if key in self.slot_of:
                continue
            slot = self.free.pop()
            self.slot_of[key] = slot
            todo.append((key, slot))
        n_detail = len(self._detail)
        t0 = time.perf_counter()
        if self._pool is None:
            for k, s in todo:
                self._load_one(k, s)
        else:
            list(self._pool.map(lambda ks: self._load_one(*ks), todo))
        del self._detail[n_detail:]          # startup loads are reported apart
        return (time.perf_counter() - t0) * 1000

    def ensure(self, keys, scores=None, sync=None, protect=()):
        """Make ``keys`` resident; returns their slots."""
        slots, misses = self.plan(keys, scores, protect)
        self.load(misses, sync)
        return slots

    def close(self):
        """Release the bank now (the model may still hold a reference to this object)."""
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
        self.GU = self.D = None
        self._stages.clear()
        self._tls = threading.local()
        self.slot_of.clear()
        self.free = []
        free_memory()


def start_pool(cache: ExpertCache, threads: int) -> ThreadPoolExecutor:
    """A reader pool whose every thread has already allocated its pinned staging buffer
    and copy stream, so that allocation is never inside a timed load."""
    pool = ThreadPoolExecutor(threads)
    barrier = threading.Barrier(threads)

    def warm():
        cache._stage()
        barrier.wait()

    for f in [pool.submit(warm) for _ in range(threads)]:
        f.result()
    return pool


def replay_two_tier(events, policy: _Policy, host_capacity: int, host_policy: str = "LRU"):
    """Exact replay of a VRAM policy plus a RAM tier, with the real cache's grouping
    and pinning rules. Returns (VRAM hit per access, RAM hit per VRAM miss)."""
    host = make_policy(host_policy, host_capacity)
    K = policy.capacity
    vram, ram = [], []
    for ev in events:
        eids, mass, layer = ev["eids"], ev["mass"], ev["layer"]
        for g0 in range(0, len(eids), K):
            grp = [(layer, e) for e in eids[g0:g0 + K]]
            pinned = set(grp)
            for key, m in zip(grp, mass[g0:g0 + K]):
                hit = policy.access(key, m, pinned)[0]
                vram.append(hit)
                if not hit:
                    ram.append(host.access(key, 1.0, pinned)[0])
    return np.asarray(vram, dtype=bool), np.asarray(ram, dtype=bool)


# ─────────────────────────────────────────────────────────────────────────────
# Routing + compute: drop-in replacement for transformers' OlmoeSparseMoeBlock
# ─────────────────────────────────────────────────────────────────────────────

def _stock_style():
    """How the installed transformers computes OLMoE experts: (fused gate_up?, block
    returns (hidden, router_logits)?). transformers 5 fuses gate/up and returns a tensor;
    transformers 4 keeps them separate and returns a tuple."""
    import inspect
    from transformers.models.olmoe import modeling_olmoe as mo
    fused = hasattr(mo, "OlmoeExperts")
    src = inspect.getsource(mo.OlmoeSparseMoeBlock.forward)
    returns_tuple = "router_logits" in src.rsplit("return", 1)[-1]
    return fused, returns_tuple


class CachedMoE(nn.Module):
    """Same routing math as transformers' ``OlmoeSparseMoeBlock``; experts come from the
    global ``ExpertCache`` bank instead of a resident 64-expert tensor."""

    def __init__(self, hidden: int, num_experts: int, layer: int, cache: ExpertCache | None,
                 top_k: int = 8, norm_topk_prob: bool = False, act_fn: nn.Module | None = None,
                 fused: bool = True, returns_tuple: bool = False):
        super().__init__()
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.act_fn = act_fn if act_fn is not None else nn.SiLU()
        self.layer = layer
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.fused = fused
        self.returns_tuple = returns_tuple
        self.cache = cache
        self.observers = []
        self.route_wait_ms = 0.0

    # router (identical to transformers)
    def route(self, x_flat):
        logits = F.linear(x_flat, self.gate.weight)
        probs = F.softmax(logits, dim=-1, dtype=torch.float)
        scores, inds = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(dim=-1, keepdim=True)
        return inds, scores.to(logits.dtype)

    def routing(self, x_flat):
        """Route and bring the decision to the CPU (this waits for the GPU)."""
        inds, scores = self.route(x_flat)
        t = time.perf_counter()
        ind_np = inds.cpu().numpy()
        self.route_wait_ms += (time.perf_counter() - t) * 1000
        sc_np = scores.float().cpu().numpy()
        eids, inv = np.unique(ind_np, return_inverse=True)
        mass = np.bincount(inv.reshape(-1), weights=sc_np.reshape(-1), minlength=len(eids))
        return inds, scores, ind_np, sc_np, eids, mass

    def token_index(self, ind_np, eids, device):
        """For each expert: (token rows, top-k position) in the order torch.where gives
        them in transformers (position-major, then token), as tensors on ``device``."""
        indT = ind_np.T
        pos, tok, counts = [], [], []
        for e in eids:
            p, t = np.nonzero(indT == e)
            pos.append(p)
            tok.append(t)
            counts.append(len(t))
        both = torch.from_numpy(np.stack([np.concatenate(tok), np.concatenate(pos)]).astype(np.int64)).to(device)
        out, o = {}, 0
        for e, c in zip(eids, counts):
            out[int(e)] = (both[0, o:o + c], both[1, o:o + c])
            o += c
        return out

    def expert_out(self, x_flat, slot, tok, pos, scores):
        """One expert's weighted output for its tokens (transformers' eager loop body)."""
        cache = self.cache
        cur = x_flat[tok]
        if self.fused:
            gate, up = F.linear(cur, cache.GU[slot]).chunk(2, dim=-1)
        else:
            I = cache.D.shape[-1]
            gate, up = F.linear(cur, cache.GU[slot, :I]), F.linear(cur, cache.GU[slot, I:])
        h = F.linear(self.act_fn(gate) * up, cache.D[slot])
        return h * scores[tok, pos, None]

    def _finish(self, y, shape):
        y = y.reshape(shape)
        return (y, None) if self.returns_tuple else y

    def forward(self, x):
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])
        inds, scores, ind_np, sc_np, eids, mass = self.routing(x_flat)
        call = MoECall(self.layer, int(ind_np.shape[0]), eids.tolist(), mass.tolist(), ind_np, sc_np)
        cache = self.cache
        h0, m0, io0 = cache.hits, cache.misses, cache.io_ms
        idx = self.token_index(ind_np, eids, x_flat.device)
        y = torch.zeros_like(x_flat)
        n, K = len(eids), cache.capacity
        ngroups = 0
        for g0 in range(0, n, K):
            # More distinct experts than slots: groups of <= K, each finished before the
            # next group's loads may overwrite its slots. Expert order is unchanged.
            if g0 > 0:
                synchronize(x_flat.device)
            grp = eids[g0:g0 + K]
            slots = cache.ensure([(self.layer, int(e)) for e in grp], mass[g0:g0 + K])
            for e, s in zip(grp, slots):
                tok, pos = idx[int(e)]
                y.index_add_(0, tok, self.expert_out(x_flat, s, tok, pos, scores).to(y.dtype))
            ngroups += 1
        call.groups = ngroups
        call.hits = cache.hits - h0
        call.misses = cache.misses - m0
        call.io_ms = cache.io_ms - io0
        for obs in self.observers:
            obs(call)
        return self._finish(y, shape)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (non-expert weights only)
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_dir: Path | str | None = None, index: ExpertIndex | None = None, device=None,
               attn_implementation: str = "sdpa"):
    """Build OLMoE on the meta device (no memory), replace every MoE block with a
    ``CachedMoE``, then allocate and load ONLY the non-expert weights (attention, norms,
    router, embeddings, head) on ``device``. The expert bank is never materialised.
    Call ``attach_cache`` before running the model."""
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.activations import ACT2FN

    model_dir = Path(model_dir or default_model_dir())
    index = index or ExpertIndex(model_dir)
    device = torch.device(device) if device is not None else default_device()
    config = AutoConfig.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    fused, returns_tuple = _stock_style()

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, attn_implementation=attn_implementation)
        for i, layer in enumerate(model.model.layers):
            layer.mlp = CachedMoE(config.hidden_size, config.num_experts, i, None, config.num_experts_per_tok,
                                  config.norm_topk_prob, ACT2FN[config.hidden_act], fused, returns_tuple)
    model = model.to(index.dtype).to_empty(device=device)
    # non-persistent buffers (rotary frequencies) are rebuilt for real on the device
    model.model.rotary_emb = type(model.model.rotary_emb)(config=model.config).to(device)

    params = dict(model.named_parameters())
    loaded = set()
    with torch.no_grad():
        for shard in index.shards:
            with safe_open(str(shard), framework="pt", device="cpu") as f:
                for k in index.non_expert_keys[str(shard)]:
                    if k not in params:
                        raise KeyError(f"checkpoint tensor {k} has no place in the model")
                    params[k].copy_(f.get_tensor(k))
                    loaded.add(k)
    missing = set(params) - loaded
    assert not missing, f"weights not loaded: {sorted(missing)[:5]}"
    assert not any(t.is_meta for t in [*model.parameters(), *model.buffers()])
    model.eval()
    gc.collect()
    return model, tokenizer, index


def moe_blocks(model) -> list[CachedMoE]:
    return [layer.mlp for layer in model.model.layers]


def attach_cache(model, cache: ExpertCache, observers=()):
    """Point every layer at ``cache`` and set the observers."""
    for blk in moe_blocks(model):
        blk.cache = cache
        blk.observers = list(observers)
        blk.route_wait_ms = 0.0


def detach_cache(model):
    """Drop the model's references to the current cache and observers."""
    for blk in moe_blocks(model):
        blk.cache = None
        blk.observers = []


def route_wait_ms(model) -> float:
    return sum(b.route_wait_ms for b in moe_blocks(model))


def non_expert_bytes(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# Generation (greedy, fixed length, explicit synchronisation every step)
# ─────────────────────────────────────────────────────────────────────────────

def _new_kv(model):
    from transformers import DynamicCache
    try:
        return DynamicCache(config=model.config)
    except TypeError:
        return DynamicCache()


def _forward_last(model, ids, kv, position_ids, attention_mask=None):
    """Run the decoder on ``ids`` and return the logits of the last position only."""
    out = model.model(input_ids=ids, attention_mask=attention_mask, position_ids=position_ids,
                      past_key_values=kv, use_cache=True)
    return model.lm_head(out.last_hidden_state[:, -1, :])


@torch.inference_mode()
def generate(model, tokenizer, prompt_tokens, max_new_tokens: int = 32,
             prefill_chunk: int | None = None, on_step=None) -> GenResult:
    """Greedy decoding for exactly ``max_new_tokens`` (EOS is not a stop, so every run
    does the same amount of work). ``prefill_chunk`` splits the prompt into chunks of
    that many tokens (None = whole prompt in one pass). ``on_step(phase, i)`` is called
    before each forward pass."""
    dev = model_device(model)
    toks = list(prompt_tokens)
    kv = _new_kv(model)
    chunk = prefill_chunk or len(toks)

    synchronize(dev)
    t0 = time.perf_counter()
    logits = None
    for i, s in enumerate(range(0, len(toks), chunk)):
        if on_step:
            on_step("prefill", i)
        ids = torch.tensor([toks[s:s + chunk]], device=dev)
        pos = torch.arange(s, s + ids.shape[1], device=dev)[None]
        logits = _forward_last(model, ids, kv, pos)
    out = [int(logits.argmax(dim=-1).item())]          # .item() synchronises
    ttft = time.perf_counter() - t0

    step_ms = []
    t1 = time.perf_counter()
    for i in range(max_new_tokens - 1):
        if on_step:
            on_step("decode", i)
        ts = time.perf_counter()
        ids = torch.tensor([[out[-1]]], device=dev)
        pos = torch.tensor([[len(toks) + i]], device=dev)
        logits = _forward_last(model, ids, kv, pos)
        out.append(int(logits.argmax(dim=-1).item()))
        step_ms.append((time.perf_counter() - ts) * 1000)
    decode = time.perf_counter() - t1
    return GenResult(out, tokenizer.decode(out), len(toks), ttft, decode, step_ms)


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts_tokens, max_new_tokens: int = 32, on_step=None) -> dict:
    """Greedy decoding of B prompts *together* (one forward pass per step for the whole
    batch). Prompts are left-padded and masked. Every step is synchronised. Returns
    tokens per request and timings (same keys as the Mac)."""
    dev = model_device(model)
    B = len(prompts_tokens)
    lens = [len(t) for t in prompts_tokens]
    L = max(lens)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    left = [L - n for n in lens]
    ids = torch.tensor([[pad] * lp + list(t) for lp, t in zip(left, prompts_tokens)], device=dev)
    mask = torch.tensor([[0] * lp + [1] * n for lp, n in zip(left, lens)], device=dev)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    kv = _new_kv(model)

    synchronize(dev)
    t0 = time.perf_counter()
    if on_step:
        on_step("prefill", 0)
    logits = _forward_last(model, ids, kv, pos, mask)
    tok = logits.argmax(dim=-1)
    out = [[t] for t in tok.tolist()]
    ttft = time.perf_counter() - t0

    step_ms = []
    t1 = time.perf_counter()
    last = pos[:, -1:]
    for i in range(max_new_tokens - 1):
        if on_step:
            on_step("decode", i)
        ts = time.perf_counter()
        mask = torch.cat([mask, torch.ones((B, 1), dtype=mask.dtype, device=dev)], dim=1)
        last = last + 1
        logits = _forward_last(model, tok[:, None], kv, last, mask)
        tok = logits.argmax(dim=-1)
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
        d0 = len(cache._detail)
        rw0 = route_wait_ms(model)

        def on_step(phase, j, _i=i, _d=p["domain"]):
            rec.set(prompt=_i, domain=_d, phase=phase, step=j)

        t = time.perf_counter()
        r = generate(model, tokenizer, toks, max_new_tokens, prefill_chunk, on_step=on_step)
        wall = time.perf_counter() - t
        hits, misses = cache.hits - h0, cache.misses - m0
        io_s = (cache.io_ms - io0) / 1000
        det = np.array(cache._detail[d0:], dtype=float).reshape(-1, 3)
        rows.append({
            "prompt": i, "domain": p["domain"], "prompt_tokens": r.prompt_tokens,
            "new_tokens": len(r.tokens), "ttft_s": r.ttft_s, "decode_s": r.decode_s,
            "decode_tok_s": r.decode_tok_s, "wall_s": wall,
            "hits": hits, "misses": misses, "hit_rate": hits / max(1, hits + misses),
            "GB_loaded": (cache.bytes_loaded - b0) / 1e9, "io_s": io_s,
            "disk_s": float(det[:, 1].sum()) / 1000, "h2d_s": float(det[:, 2].sum()) / 1000,
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

def reset_peak_memory(device=None):
    d = torch.device(device) if device is not None else default_device()
    if d.type == "cuda":
        torch.cuda.reset_peak_memory_stats(d)


def peak_memory_gb(device=None) -> float:
    """Peak memory the GPU allocator handed out since ``reset_peak_memory`` (NaN off CUDA)."""
    d = torch.device(device) if device is not None else default_device()
    if d.type == "cuda":
        return torch.cuda.max_memory_allocated(d) / 1e9
    return float("nan")


def memory_snapshot(device=None) -> dict:
    import psutil
    d = torch.device(device) if device is not None else default_device()
    out = {"process_rss_GB": psutil.Process().memory_info().rss / 1e9}
    if d.type == "cuda":
        free, total = torch.cuda.mem_get_info(d)
        out.update({"gpu_allocated_GB": torch.cuda.memory_allocated(d) / 1e9,
                    "gpu_peak_GB": torch.cuda.max_memory_allocated(d) / 1e9,
                    "gpu_reserved_GB": torch.cuda.memory_reserved(d) / 1e9,
                    "gpu_free_GB": free / 1e9})
    else:
        out.update({"gpu_allocated_GB": float("nan"), "gpu_peak_GB": float("nan"),
                    "gpu_reserved_GB": float("nan"), "gpu_free_GB": float("nan")})
    return out


def system_info(device=None) -> dict:
    import psutil
    import transformers
    d = torch.device(device) if device is not None else default_device()
    info = {
        "os": platform.platform(), "cpu": platform.processor(),
        "ram_GB": round(psutil.virtual_memory().total / 1e9, 1),
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "device": str(d),
        "expert_reads": ExpertReader._default_mode(),
    }
    if d.type == "cuda":
        p = torch.cuda.get_device_properties(d)
        info.update({"gpu": p.name, "vram_GB": round(p.total_memory / 1e9, 2), "cuda": torch.version.cuda})
    return info


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers (used at the top of every notebook)
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def reference_layer_output(model, index: ExpertIndex, blk: CachedMoE, x, reader: ExpertReader | None = None):
    """The same MoE layer computed the stock transformers way: an ``OlmoeSparseMoeBlock``
    with all 64 experts of that layer resident (only ONE layer, ~0.8 GB, for checking),
    using the eager expert loop."""
    from transformers.models.olmoe import modeling_olmoe as mo
    reader = reader or ExpertReader(index)
    dev = x.device
    cfg = copy.deepcopy(model.config)
    try:
        cfg._experts_implementation = "eager"
    except Exception:
        pass
    with torch.device("meta"):
        ref = mo.OlmoeSparseMoeBlock(cfg)
    ref = ref.to(index.dtype).to_empty(device=dev)
    ref.gate.weight.copy_(blk.gate.weight)
    ex = ref.experts
    for e in range(index.num_experts):
        g, u, d = reader.read_tensors(blk.layer, e)
        if hasattr(ex, "gate_up_proj"):
            ex.gate_up_proj[e].copy_(torch.cat([g, u], dim=0))
            ex.down_proj[e].copy_(d)
        else:
            ex[e].gate_proj.weight.copy_(g)
            ex[e].up_proj.weight.copy_(u)
            ex[e].down_proj.weight.copy_(d)
    out = ref(x)
    out = out[0] if isinstance(out, tuple) else out
    synchronize(dev)
    del ref
    return out


@torch.inference_mode()
def check_layer_exactness(model, index: ExpertIndex, capacity: int = 64, n_tokens: int = 37,
                          layer: int = 3, seed: int = 0) -> float:
    """Max |cached - reference| for one layer on random hidden states (should be 0)."""
    blk = moe_blocks(model)[layer]
    dev = blk.gate.weight.device
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn((1, n_tokens, index.hidden), generator=g) * 0.5).to(index.dtype).to(dev)
    ref = reference_layer_output(model, index, blk, x)
    cache = ExpertCache(index, capacity, "LRU", device=dev)
    old, old_obs = blk.cache, blk.observers
    blk.cache, blk.observers = cache, []
    try:
        got = blk(x)
        got = got[0] if isinstance(got, tuple) else got
    finally:
        blk.cache, blk.observers = old, old_obs
    diff = float((got.float() - ref.float()).abs().max().item())
    cache.close()
    del ref, got
    free_memory()
    return diff
