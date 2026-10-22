## Axis 2: Eviction Policy & Expert Locality

**Research Question:** Does expert routing exhibit strong enough temporal and spatial locality to justify complex eviction logic over simple FIFO / LRU?  
**Hypothesis:** Least Frequently Used (LFU) significantly outperforms LRU on homogeneous, sustained tasks by anchoring core "syntactic" experts. However, during multi-turn domain shifts, un-decayed LFU causes stale experts to remain resident, causing hit rates to degrade below LRU.

## Notebook

`Mac.ipynb` — Mac / MLX implementation (executed, results included).

### Method

- **Storage:** per-expert tensors stay on disk; a byte-range index is built from safetensors headers.
- **Residency:** one **global** cache keyed by `(layer, expert_id)`, capacity `K` across all 16 layers
  (`K=1024` is the full bank). Per-layer average is `K/16`.
- **Routing:** greedy generation with **exact expert weights paged from disk**, so the router sees
  real activations and records the model's authentic routing. Expert weights are never zeroed.
- **Accesses:** one lookup per **distinct expert per MoE call** (what a real layer fetches for a
  batch), not one per token — per-token counting inflates hit rates with free within-batch reuse.
- **Replay:** FIFO, LRU, LFU, and decayed LFU (half-life 256 / 1024 accesses) are replayed over the
  same stream at `K ∈ {128, 256, 512, 1024}`.
- **Validation:** the replay is checked against the real disk-backed cache for every policy
  (identical hit / miss / eviction counts, identical generated text).

### Running

Use a Python environment with `mlx`, `mlx-lm`, `safetensors`, `pandas`, `matplotlib`, `seaborn`
and `psutil` (the notebook's kernelspec is `python3.14`). Then **Run All**.

Traces are cached in `traces/` (generated, ignored by git). Delete them or set
`REUSE_TRACES=False` after changing prompts, and lower `TRACE_CACHE_K` on memory-constrained
machines (slower, same result).

### Default-run findings

- H1 is only partly supported: un-decayed LFU loses badly at small `K` (cold-start pollution),
  is ~equal at `K=256`, and edges ahead at `K=512` on the sustained workload.
- H2 is supported: un-decayed LFU falls below LRU on the domain-shift workload at every `K`, and
  loses ~10 points on the first new-domain segment.
- Decayed LFU is the best overall policy at `K=256` and `K=512` on domain shifts.
- Routing is only weakly frequency-skewed (Gini ≈ 0.42; top-64 experts ≈ 25% of accesses), which
  explains why pure frequency pinning does not pay off.

`eviction_policies_experiment.ipynb` is the earlier prototype. It zeroes expert weights while
tracing (which corrupts the residual stream and therefore routing) and simulates per token per
layer. `Mac.ipynb` supersedes it.
