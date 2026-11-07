## Axis 3: Prefill Granularity & Expert Amortization

**Research Question:** What is the optimal chunk size that maximizes expert reuse without exhausting the tight memory budget?
**Hypothesis:** As chunk size increases ($1 \to 128$), Time to First Token (TTFT) drops non-linearly because the probability of inter-token expert reuse within the chunk approaches 1. However, beyond $C = 128$, the set of unique requested experts exceeds cache capacity, forcing scratchpad memory expansion or intra-chunk evictions.

## Notebook

`Mac.ipynb`: Mac / MLX, built on the shared `experiments/moe_cache.py` (true K-slot disk-backed cache, see `experiments/README.md`).

### Method

* Two long passages written for this axis, trimmed to exactly 512 tokens.
* Sweep chunk size C = 1 ... 512 and the whole prompt, at K = 128, 256, 512, each from a cold cache.
* Outputs: TTFT, GB read during prefill, reuse ratio (assignments per load), distinct experts per layer call vs uniform routing, chunk footprint across 16 layers vs K, grouped (intra-chunk evicting) calls, scratch memory.

### Running

Run All (full configuration: ~40-60 min on an M4 / 16 GB). `MOE_SMOKE=1` runs a tiny configuration in a few minutes.
Results go to `results/` in this folder.
