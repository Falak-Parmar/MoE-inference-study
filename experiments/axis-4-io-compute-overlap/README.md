## Axis 4: Compute / I/O Concurrency & Pipeline Overlap

**Research Question:** Can the dense shared-expert computation fully mask asynchronous expert transfer latency?
**Hypothesis:** On high-bandwidth storage, dense shared-expert GEMM execution covers a significant fraction of single-expert miss latency. However, when $k$ routed experts miss simultaneously, aggregate transfer latency exceeds the shared-expert compute window, exposing direct pipeline stalls.

## Notebook

`Mac.ipynb`: Mac / MLX, built on the shared `moe` package (`src/moe/mlx_cache.py`, the true K-slot disk-backed cache; see `experiments/README.md`).

`Windows_DGPU.ipynb`: Windows dedicated GPU / PyTorch, the same experiment on `src/moe/torch_cache.py` (K experts in VRAM; misses read unbuffered from disk into pinned RAM, then copied to VRAM). Same prompts and configuration as the Mac; writes `results/windows-dgpu.json`. Here `overlap` keeps each expert's output apart and sums them in the stock order, so every mode is bit-exact (the Mac's `overlap` was not).

### Method

* **Reframed for OLMoE**, which has no shared expert: once a layer is routed, the compute that can hide its misses is the GEMMs of the experts that were hits, plus other reads in parallel.
* Modes: `sync-1`, `sync-4` (parallel reads), `overlap-1`, `overlap-4` (hit experts launched with `mx.async_eval` while misses are read).
* Every MoE call is timed and grouped by m = misses in the call (1-8); exposed stall(m) = T(m) - T(0). Plus a microbenchmark of read latency for m experts with 1-8 threads.

### Running

Run All (full configuration: ~20-30 min on an M4 / 16 GB). `MOE_SMOKE=1` runs a tiny configuration in a few minutes.
Results: `results/mac.json`, `results/windows-dgpu.json` (one file per device; format in `experiments/README.md`). Prompts: `prompts.json` (the same on every device). Figures with every device: `figures/`, made by `python -m moe.figures`.
