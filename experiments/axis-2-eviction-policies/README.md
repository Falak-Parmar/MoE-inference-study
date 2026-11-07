## Axis 2: Eviction Policy & Expert Locality

**Research Question:** Does expert routing exhibit strong enough temporal and spatial locality to justify complex eviction logic over simple FIFO / LRU?
**Hypothesis:** Least Frequently Used (LFU) significantly outperforms LRU on homogeneous, sustained tasks by anchoring core "syntactic" experts. However, during multi-turn domain shifts, un-decayed LFU causes stale experts to remain resident, causing hit rates to degrade below LRU.

## Notebook

`Mac.ipynb`: Mac / MLX, built on the shared `experiments/moe_cache.py` (true K-slot disk-backed cache, see `experiments/README.md`).

### Method

* Policies: FIFO, LRU, LFU, decayed LFU (half-life 1024 / 4096 accesses) and the new **router-score-aware (RSA)** policy, which evicts the expert with the smallest decayed sum of router-gate probability mass.
* Streams: sustained (12 programming requests) and domain shift (programming -> creative -> technical -> math, 4 each), extending the team's original prompts.
* Every policy runs on the real disk-backed cache at K = 256 (timed; identical text and routing asserted). The K sweep (128-512) uses exact replay of those streams.
* Outputs: hit rate vs K, post-shift and per-segment hit rates, rolling hit rate, real I/O time and tok/s per policy, locality analysis (frequency skew, reuse time, top-256 domain overlap), H1/H2 checks.

### Running

Run All (full configuration: ~1.5-2 h on an M4 / 16 GB). `MOE_SMOKE=1` runs a tiny configuration in a few minutes.
Results go to `results/` in this folder.

### Earlier results (previous notebook, superseded)

The previous Axis 2 notebook (in git history) reported, on its own prompt set: un-decayed LFU loses badly at
small K (cold-start pollution), roughly ties LRU at K=256 and edges ahead at K=512 on the sustained workload;
un-decayed LFU falls below LRU on the domain-shift workload at every K; decayed LFU was best at K=256-512 on
domain shifts; routing was only weakly frequency-skewed (Gini about 0.42). The rebuilt notebook re-tests these
claims with the new cache, new prompts and the RSA policy.
