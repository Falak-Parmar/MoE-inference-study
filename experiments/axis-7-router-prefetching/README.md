## Axis 7: Router-Guided Speculative Prefetching

**Research Question:** If the router's logits for layer N are computed slightly ahead of when layer N's experts are actually needed, can that lookahead window be used to issue prefetch commands before the compute engine stalls on a cache miss?
**Hypothesis:** Speculative prefetching in the window between "router decision known" and "expert needed" reduces transfer wait for correctly predicted experts, but mispredictions waste bandwidth, creating a precision/recall trade-off between prefetch aggressiveness and wasted I/O.

## Notebook

`Mac.ipynb`: Mac / MLX, built on the shared `moe` package (`src/moe/mlx_cache.py`, the true K-slot disk-backed cache; see `experiments/README.md`).

### Method

* Predictor: apply layer L+d's router to layer L's hidden state (RMSNorm weight swapped), d = 1, 2, 3; baseline = the experts the same layer used for the previous token.
* Real background prefetching into cache slots (in-flight slots protected from eviction), varying P (experts per token), lookahead d and reader threads. Text is asserted identical in every mode.
* Outputs: offline precision/recall vs P and d, per-layer recall, real stall time, useful vs wasted prefetches, decode tok/s and TTFT.

### Running

Run All (full configuration: ~25-35 min on an M4 / 16 GB). `MOE_SMOKE=1` runs a tiny configuration in a few minutes.
Results: `results/mac.json` (one file per device; format in `experiments/README.md`). Prompts: `prompts.json` (the same on every device). Figures with every device: `figures/`, made by `python -m moe.figures`.
