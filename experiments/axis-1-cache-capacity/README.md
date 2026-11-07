## Axis 1: Cache Capacity vs Working Set

**Research Question:** At what cache capacity does the working set saturate, and when does adding memory yield diminishing returns?
**Hypothesis:** At low cache sizes ($K < 8$), the system is purely I/O bound. Beyond a critical knee ($K \approx 24$), the hit rate plateaus because tail experts are rarely activated. At this saturation threshold, the primary performance bottleneck abruptly shifts from storage/bus bandwidth to GPU compute.

## Notebook

`Mac.ipynb`: Mac / MLX, built on the shared `experiments/moe_cache.py` (true K-slot disk-backed cache, see `experiments/README.md`).

### Method

* K is **global** over 1,024 logical experts; the hypothesis numbers are per layer, so they map to global K = 128 (8/layer) and 384 (24/layer).
* Sweep K = 64, 96, 128, 192, 256, 384, 512, 768, 1024 on 12 mixed requests (math, code, chat, writing) x 64 new tokens, LRU.
* Real timed runs for every K whose bank fits in RAM; exact replay of the recorded access stream for the rest (replay is asserted equal to every real run).
* Outputs: hit rate (overall / prefill / decode) vs K with the knee, decode tok/s, TTFT, I/O vs non-I/O time, working set W(tau), cold vs steady state, per-layer slot occupancy.

### Running

Run All (full configuration: ~30-60 min on an M4 / 16 GB). `MOE_SMOKE=1` runs a tiny configuration in a few minutes.
Results go to `results/` in this folder.
