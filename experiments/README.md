# Experiments

One folder per research axis. Axis 6 (mixed-precision tiering) was cancelled. Each folder has:

```
axis-N-name/
  README.md
  Mac.ipynb                 # Apple Silicon, MLX
  Windows_IGPU.ipynb        # Windows, integrated / shared-memory GPU, PyTorch (to come)
  Windows_DGPU.ipynb        # Windows, dedicated GPU + VRAM, PyTorch
  prompts.json              # the axis's workload: the SAME prompts on every device
  results/
    mac.json                # everything one device measured (one file per device)
    windows-igpu.json
    windows-dgpu.json
  figures/*.png             # report figures, every device on the same graph
```

| Axis | Folder | Question |
|---|---|---|
| 1 | `axis-1-cache-capacity` | At what cache capacity K does the working set saturate? |
| 2 | `axis-2-eviction-policies` | Is routing local enough to justify smarter eviction than LRU / FIFO? |
| 3 | `axis-3-prefill-chunking` | Which prefill chunk size amortises expert loads best? |
| 4 | `axis-4-io-compute-overlap` | How much expert-read latency can compute hide? |
| 5 | `axis-5-memory-tiering` | Which experts should live in unified memory, and should that set change? |
| 7 | `axis-7-router-prefetching` | Can the router predict upcoming experts well enough to prefetch them? |
| 8 | `axis-8-batch-contention` | Does batching destroy the locality an expert cache relies on? |

## Shared code

All shared code is the `moe` package in `src/moe/`. Install it once per machine from the repository
root: `pip install -e ".[mac]"` on the Mac, `pip install -e ".[windows]"` on Windows. The notebooks
also add `src/` to `sys.path` themselves, so they run without installing.

| Module | Used by | Needs |
|---|---|---|
| `moe.common` | every device | numpy |
| `moe.mlx_cache` | Mac notebooks | MLX (Mac only) |
| `moe.torch_cache` | Windows notebooks (`Windows_DGPU.ipynb`) | PyTorch, transformers |
| `moe.results` | every notebook, all devices | pandas |
| `moe.figures` | every notebook, and run once for the report | pandas, matplotlib |

`moe.common` has everything that does not depend on a framework:
* the safetensors byte-range index
* the eviction policies (FIFO, LRU, LFU, RSA, tiered)
* exact `replay`
* the call `Recorder`
* the workload helpers
* the working-set, knee and Gini analysis

Every device uses these same objects, so policies and replay cannot drift apart between
devices. `moe.mlx_cache` re-exports all of `moe.common`, so in a Mac notebook `mc.make_policy`,
`mc.replay` and the rest work as before.

Imports in a notebook:

```python
from moe import mlx_cache as mc, results as rs, figures as fg     # Mac
from moe import torch_cache as tc, results as rs, figures as fg   # Windows
```

### `moe.mlx_cache` (Mac)

Every Mac notebook builds on this module. It keeps storage, residency, routing and compute separate:

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

### `moe.torch_cache` (Windows dGPU)

Windows with a CUDA GPU only (it uses the Win32 API for reads and fails clearly elsewhere). The
PyTorch counterpart of `moe.mlx_cache`, with the same names and arguments, so a
`Windows_DGPU.ipynb` is its axis's `Mac.ipynb` with `mc` replaced by `tc`. Policies, replay and the
recorder are the same `moe.common` objects.

* **Storage** — the same `ExpertIndex`. `ExpertReader` reads each projection unbuffered
  (`FILE_FLAG_NO_BUFFERING` through `CreateFileW` / `ReadFile`), so the Windows file cache never
  holds expert bytes and no purge is needed. Byte
  ranges are widened to 4 KiB for unbuffered I/O. If the unbuffered open fails, it falls back to
  buffered reads and reports `mode = 'buffered'`.
* **Residency** — `ExpertCache` is a bank of **K global slots in VRAM** (`GU` = [gate; up] and `D`).
  A miss goes disk → pinned staging buffer (one per reader thread) → async copy on that thread's
  CUDA stream into the slot. Disk-read and host → VRAM copy times are recorded separately
  (`disk_s`, `h2d_s`). `HostTier` adds a bounded pinned-RAM tier between VRAM and disk (Axis 5);
  `replay_two_tier` replays both tiers exactly.
* **Routing + compute** — `CachedMoE` replaces each `OlmoeSparseMoeBlock`: the same router math as
  transformers, then transformers' own eager per-expert loop (same expert order, same gather, gate/up
  fused or split as the installed transformers version does it, same `index_add_`) over the VRAM
  slots. It is checked bit-exact against the stock transformers layer (all 64 experts resident,
  eager experts) at the top of every notebook.
* **Measurement** — greedy single and batched generation (`DynamicCache`, left padding with an
  attention mask), every step synchronised; `max_safe_capacity` reads the free VRAM once at load;
  `system_info` records the GPU, VRAM, CUDA, torch and transformers versions and the read mode.

Each axis has its own prompts (per-axis workloads) in `prompts.json`; nothing is downloaded by the
notebooks.

## Running

1. Python 3.14 environment with `mlx`, `mlx-lm`, `safetensors`, `numpy`, `pandas`, `matplotlib`,
   `psutil` (kernel `python3.14`).
2. Model files in `models/OLMoE-1B-7B-0125-Instruct/` (F16 safetensors).
3. Open a notebook and **Run All**. It writes `results/mac.json` and shows the cross-device figures.
   It does not write PNGs (see [Figures](#figures)).

Quick check of any notebook (tiny configuration, a few minutes):

```bash
MOE_SMOKE=1 jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3.14 --output /tmp/smoke.ipynb experiments/axis-1-cache-capacity/Mac.ipynb
```

Smoke results go to `experiments/.cache/smoke/` (git-ignored).

**Windows (dGPU).**

1. Python environment with a CUDA build of PyTorch, then `pip install -e ".[windows]"` from the
   repository root (adds transformers, safetensors, pandas, matplotlib, psutil).
2. Download `allenai/OLMoE-1B-7B-0125-Instruct` (BF16, ~14 GB) into
   `models/OLMoE-1B-7B-0125-Instruct/` (git-ignored) beforehand. It is the same checkpoint as the
   Mac's, so routing is comparable. The notebooks stop with a clear message if it is missing.
3. Close other GPU programs (each notebook reads the free VRAM once at load and times only the K
   values whose bank fits), open `Windows_DGPU.ipynb` and **Run All**. It writes
   `results/windows-dgpu.json`.

Quick check (`MOE_SMOKE=1`, from a Command Prompt; in PowerShell use `$env:MOE_SMOKE=1`):

```bash
set "MOE_SMOKE=1" && jupyter nbconvert --to notebook --execute --output-dir experiments/.cache/smoke experiments/axis-1-cache-capacity/Windows_DGPU.ipynb
```

**Cold disk.** `F_NOCACHE` stops these reads from filling the page cache, but data already cached is
still served from RAM. Notebooks try `sudo -n purge`; if that needs a password, run `sudo purge`
yourself before a full run.

**Memory.** On a 16 GB Mac, banks above about 10 GB (K > ~830) are not timed; their hit rates come
from exact replay instead.

## Results: one file per device

A run of any notebook on any device ends with one call:

```python
import results as rs
rs.save(RESULTS_DIR, AXIS, DEVICE, TABLES, config=CONFIG, prompts=PROMPT_FILE, system=SYSTEM)
# -> experiments/<axis>/results/<device>.json
```

* `DEVICE` is `mac`, `windows-igpu` or `windows-dgpu`. Each device commits **only its own file**, so
  teammates never edit the same file, and re-running one device replaces only that device's results.
* `TABLES` maps a table name to a pandas DataFrame. Each table is **long / tidy**: one row per
  measurement, one column per variable, plain units (seconds, ms, GB, rates in 0-1).
* `config` is the run's settings (`rs.config_from(globals())` after the config cell collects every
  UPPER_CASE value), `prompts` is `rs.load_prompts(AXIS_DIR)`, and `system` is the machine
  (chip / GPU, RAM / VRAM, OS, framework versions).
* `rs.save` refuses a file that lacks a table or column the figures need (listed below and in
  `results.REQUIRED`); `rs.check(AXIS, TABLES)` lists what is missing before saving. Extra tables and
  extra columns are welcome; figures ignore them.
* To read results back: `rs.load(results_dir)` gives `{device: document}`, and
  `rs.table(results_dir, 'summary')` stacks one table from every device with a `device` column.

The file is JSON:
`{"schema", "axis", "device", "created", "git_commit", "system", "config", "prompts", "notes",
"tables": {name: {"columns": [...], "data": [[...], ...]}}}`.

### What every device must record

Use the same names, units and meanings on every device. Run the same configuration as the Mac
notebook where the hardware allows it (same K values, chunk sizes, batch sizes, token counts), and
use the prompts in `prompts.json` unchanged.

**K on Windows.** K is the number of experts in the fastest tier: VRAM on the dGPU, the shared GPU
memory on the iGPU. Record anything about the RAM tier as extra columns (for example
`ram_hit_rate`).

| Axis | Table | Required columns (units) |
|---|---|---|
| 1 | `summary` (one row per K) | `K`, `hit_all`, `hit_decode`, `hit_prefill` (replay, every K), `decode_tok_s`, `ttft_mean_s`, `io_s`, `wall_s`, `io_share` = io_s / wall_s, `ms_per_load` (ms per expert read). Timing columns are empty (NaN) for K that were not timed. |
| 1 | `per_request` (one row per K and request) | `K`, `prompt` (0 = first request, cold cache), `hits`, `misses`, `hit_rate` |
| 1 | `working_set` | `tau` (window in generated tokens), `W` (distinct (layer, expert) in the window, decode) |
| 2 | `sweep` (replay; one row per stream, K, policy) | `stream` (`sustained` / `domain_shift`), `K`, `policy`, `hit_rate`, `decode_hit_rate`, `post_shift` (domain_shift only: hit rate over the first decode accesses after each switch) |
| 2 | `real_runs` (one row per stream, policy) | `stream`, `policy`, `K`, `hit_rate`, `io_s`, `decode_tok_s`, `ttft_mean_s` |
| 3 | `runs` (one row per K, chunk, prompt) | `K`, `C` (chunk size or `full`), `C_num` (tokens per pass), `prompt`, `ttft_s`, `GB_loaded`, `scratch_GB` (peak memory - expert bank - other weights), `distinct_per_call` (distinct experts per layer call), `chunk_footprint` (experts one chunk needs, summed over layers) |
| 4 | `micro_reads` (one row per repetition) | `threads`, `m` (experts read together), `ms` |
| 4 | `calls_by_m` (one row per mode, phase, m) | `mode`, `phase` (`prefill` / `decode`), `m` (misses in the MoE call), `calls`, `t_moe_ms` (median MoE-layer time) |
| 4 | `runs` (one row per mode) | `mode`, `decode_tok_s`, `io_s` |
| 5 | `runs` (timed; one row per budget, strategy) | `B` (experts in fast memory), `strategy` (`static` / `hybrid` / `dynamic`), `hit_rate`, `GB_loaded`, `decode_tok_s`, `ttft_mean_s`, `startup_s` (preloading pinned experts) |
| 5 | `replay` | `B`, `strategy`, `hit_rate`, `GB_read` |
| 7 | `prediction` (one row per phase, predictor, lookahead, P, layer) | `phase`, `predictor`, `lookahead` (0 = previous-token baseline), `P` (experts predicted), `layer`, `n` (calls averaged), `precision`, `recall` |
| 7 | `runs` (one row per prefetch mode) | `mode`, `P`, `lookahead`, `threads`, `decode_tok_s`, `demand_load_s`, `prefetch_wait_s`, `stall_s` = demand + wait, `useful`, `wasted` (prefetched experts) |
| 8 | `runs` (one row per K, batch size) | `K`, `B`, `agg_tok_s` (all requests), `per_req_tok_s`, `decode_hit_rate`, `loads_per_step`, `reloads_per_step` (thrash), `step_footprint` (cache lookups per step), `step_p50_ms` |

Mode, policy and strategy labels are free text, but use the Mac labels for the same setting
(`sync-1`, `overlap-4`, `LRU`, `RSA-H1024`, `P8-d2`, ...) so bars line up.

## Figures

**Colour is always the device:** Mac blue, Windows iGPU orange, Windows dGPU green. The second
variable (K, policy, strategy, mode) is a line style, a marker or its own panel. Every figure says in
its corner which devices it contains.

| Axis | Figure | x | y | Series / panels | From |
|---|---|---|---|---|---|
| 1 | `hit_rate_vs_K` | K (log2) | hit rate | solid = decode, dotted = prefill; device knee (vertical line); hypotheses K = 128 and 384 | `summary` |
| 1 | `time_vs_K` | K, timed only | 4 panels: decode tok/s, mean TTFT, I/O share, ms per expert read | device | `summary` |
| 1 | `working_set` | τ (log2) | W(τ) | device; 128 = one token | `working_set` |
| 1 | `cold_vs_steady` | K | hit rate | solid = steady (requests 1+), dashed = cold (request 0) | `per_request` |
| 2 | `hit_vs_K` | K | hit rate | colour = policy; columns = device; rows = sustained, domain shift, post-shift | `sweep` |
| 2 | `real_cost` | policy | decode tok/s, disk-read time | grouped bars by device; rows = stream | `real_runs` |
| 3 | `chunking` | C (log2, then `full`) | 3 panels: TTFT (log), GB read, scratch GB | device; line style = K; mean over prompts | `runs` |
| 3 | `distinct` | C | distinct experts per layer call; chunk footprint | device; uniform-routing curve; K lines | `runs` |
| 4 | `read_latency` | m | median ms | device; panel per thread count | `micro_reads` |
| 4 | `moe_time` | m | median MoE-layer ms (decode) | device; panel per mode | `calls_by_m` |
| 4 | `throughput` | mode | decode tok/s; I/O time | grouped bars by device | `runs` |
| 5 | `hit_rate` | B | hit rate | device; line = replay, star = timed run; panel per strategy | `replay`, `runs` |
| 5 | `timing` | B | rows: decode tok/s, TTFT, startup s | device; column per strategy | `runs` |
| 7 | `prediction` | P | recall (solid), precision (dotted), decode | device; dashed = previous-token recall; panel per lookahead | `prediction` |
| 7 | `prefetch_cost` | mode | decode tok/s; stall s; share of prefetches wasted | grouped bars by device | `runs` |
| 8 | `batching` | B (log2) | 4 panels: aggregate tok/s, decode hit rate, loads per step, lookups per step | device; line style = K; K and no-sharing reference lines | `runs` |

**Making them.** Notebooks show these figures for every device whose file is in `results/`, but do
not save them. Once all devices have committed their results, run this from the repository root. It
writes `experiments/<axis>/figures/<name>.png`:

```bash
python -m moe.figures
```

To draw only some axes:

```bash
python -m moe.figures axis-1-cache-capacity axis-8-batch-contention
```

(Without `pip install -e .`, prefix the command with `PYTHONPATH=src`.)

Re-run it whenever a results file changes, and commit the PNGs with the results that made them.
A new figure is one function in `src/moe/figures.py` reading tables from `rs.table(...)`. Add its columns to
`moe.results.REQUIRED` and to the tables above.

**Reading them across devices.** The Mac runs MLX with F16 experts; Windows runs PyTorch with BF16.
They are different implementations, so compare the **shape of each device's curve** (where the
knee is, which policy wins, where batching starts to thrash) rather than absolute numbers alone.
Routing-derived metrics (hit rates by replay, working set, footprints, prediction accuracy) depend on
the model and prompts, not the hardware. They should nearly coincide across devices, and a gap there
points to a precision or implementation difference worth reporting.

The current `results/mac.json` files come from the 2026-11-07 full run (commit 918204b). They were
converted from that run's CSV files; the notebooks' saved outputs are from the same run.
