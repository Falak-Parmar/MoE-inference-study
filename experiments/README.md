# Experiments

One folder per research axis. Each has a `Mac.ipynb` (Apple Silicon, MLX) and a `README.md`.
Axis 6 (mixed-precision tiering) was cancelled.

| Axis | Folder | Question |
|---|---|---|
| 1 | `axis-1-cache-capacity` | At what cache capacity K does the working set saturate? |
| 2 | `axis-2-eviction-policies` | Is routing local enough to justify smarter eviction than LRU / FIFO? |
| 3 | `axis-3-prefill-chunking` | Which prefill chunk size amortises expert loads best? |
| 4 | `axis-4-io-compute-overlap` | How much expert-read latency can compute hide? |
| 5 | `axis-5-memory-tiering` | Which experts should live in unified memory, and should that set change? |
| 7 | `axis-7-router-prefetching` | Can the router predict upcoming experts well enough to prefetch them? |
| 8 | `axis-8-batch-contention` | Does batching destroy the locality an expert cache relies on? |

## Shared code: `moe_cache.py`

Every Mac notebook imports this one module. It keeps storage, residency, routing and compute separate:

* **Storage** — `ExpertIndex` reads only the safetensors headers and records the byte range of every
  expert projection. `ExpertReader` reads an expert with `pread` and `F_NOCACHE`, so reads do not
  fill the OS page cache.
* **Residency** — `ExpertCache` is a preallocated bank of **K global slots** over the 1,024 logical
  `(layer, expert_id)` experts. A miss reads the expert's bytes **straight from disk into its slot**
  through a zero-copy view of the MLX buffer. Resident expert memory is exactly K x 12.6 MB (F16);
  the full 12.9 GB bank is never loaded.
* **Policies** — FIFO, LRU, LFU (optionally decayed), the new router-score-aware policy (RSA) and a
  tiered policy (pinned static tier + dynamic tier). Keys needed by the current call are pinned.
* **Routing + compute** — `CachedMoE` replaces each OLMoE MoE block: same router math as mlx-lm, then
  `mx.gather_mm` over the slot bank. It is checked bit-exact against the stock mlx-lm layer at the
  top of every notebook.
* **Measurement** — greedy single and batched generation with every step synchronised, a recorder
  of every MoE call, and `replay`, which reproduces the real cache's hit / miss / eviction counts
  exactly (asserted in the notebooks).

Each notebook writes its own prompts inline (per-axis workloads); nothing is downloaded.

## Running

1. Python 3.14 environment with `mlx`, `mlx-lm`, `safetensors`, `numpy`, `pandas`, `matplotlib`,
   `psutil` (kernel `python3.14`).
2. Model files in `models/OLMoE-1B-7B-0125-Instruct/` (F16 safetensors).
3. Open a notebook and **Run All**. Results (CSV, JSON, PNG) go to the axis folder's `results/`.

Quick check of any notebook (tiny configuration, a few minutes):

```bash
MOE_SMOKE=1 jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3.14 --output /tmp/smoke.ipynb experiments/axis-1-cache-capacity/Mac.ipynb
```

Smoke results go to `experiments/.cache/smoke/` (git-ignored).

**Cold disk.** `F_NOCACHE` stops these reads from filling the page cache, but data already cached is
still served from RAM. Notebooks try `sudo -n purge`; if that needs a password, run `sudo purge`
yourself before a full run.

**Memory.** On a 16 GB Mac, banks above about 10 GB (K > ~830) are not timed; their hit rates come
from exact replay instead.
