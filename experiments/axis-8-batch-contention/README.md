## Axis 8: Batch Size vs Cache Contention

**Research Question:** As batch size increases, more distinct experts are likely to be activated simultaneously across different sequences in the batch. Does this destroy the locality assumptions that make caching effective?
**Hypothesis:** Small batches preserve strong locality, giving high hit rates. As batch size grows, the union of experts needed per step grows faster than cache capacity, causing thrashing that could make throughput worse than smaller batches despite better GPU utilization, implying an optimal batch size exists.

## Notebook

`Mac.ipynb`: Mac / MLX, built on the shared `moe` package (`src/moe/mlx_cache.py`, the true K-slot disk-backed cache; see `experiments/README.md`).

`Windows_DGPU.ipynb`: Windows dedicated GPU / PyTorch, the same experiment on `src/moe/torch_cache.py` (K experts in VRAM; misses read unbuffered from disk into pinned RAM, then copied to VRAM). Same prompts and configuration as the Mac; writes `results/windows-dgpu.json`.

`Windows_IGPU.ipynb`: Windows Intel integrated GPU / PyTorch (`xpu`), the same experiment on the same `src/moe/torch_cache.py` (K experts in shared GPU memory, i.e. system RAM the iGPU may use; misses read unbuffered from disk, then copied into the GPU's allocation). Same prompts and configuration as the Mac; K values whose bank does not fit in the free shared memory are replay-only. Writes `results/windows-igpu.json`.

### Method

* **Real batched decoding** (no simulation): B = 1, 2, 4, 8, 16 left-padded requests through mlx-lm `BatchKVCache`, at K = 256 and 512.
* 16 short mixed requests written for this axis (similar length to keep padding small).
* Outputs: aggregate and per-request tok/s, step p50/p95, expert union per layer and per step, decode hit rate, reloads per step (thrash), and token agreement with single-request decoding.

### Running

Run All (full configuration: ~30-45 min on an M4 / 16 GB). `MOE_SMOKE=1` runs a tiny configuration in a few minutes.
Results: `results/mac.json`, `results/windows-igpu.json`, `results/windows-dgpu.json` (one file per device; format in `experiments/README.md`). Prompts: `prompts.json` (the same on every device). Figures with every device: `figures/`, made by `python -m moe.figures`.
