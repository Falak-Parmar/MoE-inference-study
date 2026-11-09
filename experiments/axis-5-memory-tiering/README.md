## Axis 5: Memory Hierarchy & Expert Placement

**Research Question:** How does expert placement across GPU VRAM, system RAM, and storage affect inference latency and throughput?
**Hypothesis:** A multi-tier hierarchical cache outperforms single-tier caching by keeping frequently reused experts in the fastest tier, staging warm experts in an intermediate tier, and falling back to disk storage only for cold tails.

## Notebook

`Mac.ipynb`: Mac / MLX, built on the shared `moe` package (`src/moe/mlx_cache.py`, the true K-slot disk-backed cache; see `experiments/README.md`).

`Windows_DGPU.ipynb`: Windows dedicated GPU / PyTorch, the same experiment on `src/moe/torch_cache.py` (K experts in VRAM; misses read unbuffered from disk into pinned RAM, then copied to VRAM). Same prompts and configuration as the Mac; writes `results/windows-dgpu.json`. It has three tiers: the VRAM strategies are run with a pinned-RAM tier of `RAM_TIER` experts between VRAM and disk (`runs`) and without it (`runs_two_tier`).

### Method

* **Mac has two tiers: unified memory and disk** (Windows has three: VRAM, RAM, disk).
* At equal unified-memory budgets B = 192, 384, 576 experts, compare **static** (hottest experts from a profiling run pinned + 64 staging slots), **hybrid** (half pinned, half LRU) and **dynamic** (all LRU).
* Profiling and evaluation prompts are disjoint. Real timed runs plus an exact-replay budget curve.

### Running

Run All (full configuration: ~45-60 min on an M4 / 16 GB). `MOE_SMOKE=1` runs a tiny configuration in a few minutes.
Results: `results/mac.json`, `results/windows-dgpu.json` (one file per device; format in `experiments/README.md`). Prompts: `prompts.json` (the same on every device). Figures with every device: `figures/`, made by `python -m moe.figures`.
