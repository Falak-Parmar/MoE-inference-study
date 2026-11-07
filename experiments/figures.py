"""Cross-device figures: every graph puts all devices that have results side by side.

Reads ``experiments/<axis>/results/<device>.json`` (see results.py) and draws the
figures listed in experiments/README.md. Colour always means device; the second
variable (K, policy, strategy, mode) is a line style or its own panel.

    python experiments/figures.py                 # every axis -> <axis>/figures/*.png
    python experiments/figures.py axis-1-cache-capacity

In a notebook: ``figures.make(AXIS, RESULTS)`` shows them without saving.
Only pandas, numpy and matplotlib are needed (no MLX / PyTorch).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import results as rs  # noqa: E402

EXPERIMENTS = Path(__file__).parent

# Device colours (first three categorical slots: distinguishable for colour-blind readers).
COLORS = {"mac": "#2a78d6", "windows-igpu": "#eb6834", "windows-dgpu": "#1baf7a"}
# Colours for up to eight non-device series (e.g. Axis 2 policies), in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D"]

STYLE = {"figure.dpi": 100, "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
         "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
         "lines.markersize": 6, "legend.frameon": False, "legend.fontsize": 8}


def _label(device):
    return rs.DEVICES.get(device, device)


def _log2_x(ax, ticks):
    ticks = sorted(set(int(t) for t in ticks))
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()


def _device_lines(ax, df, x, y, style="-", marker="o", label_suffix="", **kw):
    """One line per device, coloured by device."""
    for dev, d in df.groupby("device", sort=False):
        d = d.dropna(subset=[y]).sort_values(x)
        if len(d):
            ax.plot(d[x], d[y], linestyle=style, marker=marker, color=COLORS.get(dev, "grey"),
                    label=_label(dev) + label_suffix, **kw)


def _grouped_bars(ax, df, x, y, order=None):
    """Bars grouped by ``x``, one bar per device, coloured by device."""
    order = order or list(dict.fromkeys(df[x]))
    devices = list(dict.fromkeys(df.device))
    width = 0.8 / max(1, len(devices))
    for i, dev in enumerate(devices):
        d = df[df.device == dev].set_index(x).reindex(order)
        pos = np.arange(len(order)) + (i - (len(devices) - 1) / 2) * width
        ax.bar(pos, d[y].to_numpy(dtype=float), width * 0.92, color=COLORS.get(dev, "grey"),
               label=_label(dev))
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.grid(axis="x", visible=False)


def _knee(x, y):
    """Kneedle knee on a log2 x axis (same rule as moe_cache.knee_point)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None
    xs = np.log2(x)
    xn = (xs - xs[0]) / (xs[-1] - xs[0])
    yn = (y - y[0]) / ((y[-1] - y[0]) or 1.0)
    return x[int(np.argmax(yn - xn))]


def _devices_note(docs):
    return "devices: " + ", ".join(_label(d) for d in docs)


# ─────────────────────────────────────────────────────────────────────────────
# One function per axis: docs (from results.load) -> {figure name: Figure}
# ─────────────────────────────────────────────────────────────────────────────

def axis1(docs):
    figs = {}
    s = rs.table(None, "summary", docs)
    if len(s):
        fig, ax = plt.subplots(figsize=(11, 5))
        _device_lines(ax, s, "K", "hit_decode", "-", "o", " · decode")
        _device_lines(ax, s, "K", "hit_prefill", ":", "^", " · prefill", alpha=0.7)
        for dev, d in s.groupby("device", sort=False):
            k = _knee(d.sort_values("K").K, d.sort_values("K").hit_decode)
            if k is not None:
                ax.axvline(k, color=COLORS.get(dev, "grey"), lw=1, alpha=0.5)
                ax.text(k, 0.97, f" knee K={k:g}", color="#555", fontsize=8, va="top")
        for k, txt in [(128, "8 / layer"), (384, "24 / layer")]:
            ax.axvline(k, color="#999", ls="--", lw=1)
            ax.text(k, 0.03, f" hypothesis {txt}", color="#555", fontsize=8)
        _log2_x(ax, s.K)
        ax.set_ylim(0, 1)
        ax.set_xlabel("global K (resident experts; per layer = K / 16)")
        ax.set_ylabel("hit rate")
        ax.set_title("Hit rate vs cache capacity")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
        figs["hit_rate_vs_K"] = fig

        t = s.dropna(subset=["decode_tok_s"])
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        panels = [("decode_tok_s", "Decode throughput", "tokens / s"),
                  ("ttft_mean_s", "Mean time to first token", "s"),
                  ("io_share", "Share of time spent reading experts", "I/O time / wall time"),
                  ("ms_per_load", "Read time per expert", "ms")]
        for ax, (col, title, yl) in zip(axes.flat, panels):
            _device_lines(ax, t, "K", col)
            _log2_x(ax, t.K)
            ax.set_title(title); ax.set_ylabel(yl); ax.set_xlabel("global K")
        axes[1, 0].set_ylim(0, 1)
        axes[0, 0].legend()
        fig.suptitle("Timed runs vs cache capacity")
        fig.tight_layout()
        figs["time_vs_K"] = fig

    w = rs.table(None, "working_set", docs)
    if len(w):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        _device_lines(ax, w, "tau", "W")
        ax.axhline(128, color="#999", ls="--", lw=1)
        ax.text(w.tau.min(), 128, " one token needs 8 x 16 = 128", color="#555", fontsize=8, va="bottom")
        _log2_x(ax, w.tau)
        ax.set_xlabel("window τ (generated tokens)"); ax.set_ylabel("distinct (layer, expert) in window")
        ax.set_title("Decode working set W(τ)"); ax.legend()
        figs["working_set"] = fig

    p = rs.table(None, "per_request", docs)
    if len(p):
        rows = []
        for (dev, K), d in p.groupby(["device", "K"], sort=False):
            warm = d[d.prompt > 0]
            rows.append({"device": dev, "K": K, "cold": d[d.prompt == 0].hit_rate.mean(),
                         "steady": warm.hits.sum() / (warm.hits.sum() + warm.misses.sum())})
        cs = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        _device_lines(ax, cs, "K", "steady", "-", "o", " · steady (requests 1+)")
        _device_lines(ax, cs, "K", "cold", "--", "s", " · cold (request 0)", alpha=0.8)
        _log2_x(ax, cs.K)
        ax.set_ylim(0, 1); ax.set_xlabel("global K"); ax.set_ylabel("hit rate")
        ax.set_title("Cold-start vs steady-state hit rate (timed runs)"); ax.legend()
        figs["cold_vs_steady"] = fig
    return figs


def axis2(docs):
    figs = {}
    sw = rs.table(None, "sweep", docs)
    if len(sw):
        devices = list(dict.fromkeys(sw.device))
        policies = list(dict.fromkeys(sw.policy))
        rows = [("sustained", "hit_rate", "Sustained stream: hit rate"),
                ("domain_shift", "hit_rate", "Domain-shift stream: hit rate"),
                ("domain_shift", "post_shift", "Domain-shift: hit rate just after each switch")]
        fig, axes = plt.subplots(len(rows), len(devices), figsize=(6.5 * len(devices), 12),
                                 squeeze=False, sharey="row")
        for j, dev in enumerate(devices):
            for i, (stream, col, title) in enumerate(rows):
                ax = axes[i, j]
                d = sw[(sw.device == dev) & (sw.stream == stream)]
                for k, pol in enumerate(policies):
                    dd = d[d.policy == pol].sort_values("K")
                    if len(dd) and dd[col].notna().any():
                        ax.plot(dd.K, dd[col], "-o", color=SERIES[k % len(SERIES)], label=pol,
                                lw=2.5 if pol.startswith("RSA") else 1.5)
                _log2_x(ax, sw.K)
                ax.set_title(f"{_label(dev)}\n{title}" if i == 0 else title)
                ax.set_xlabel("global K"); ax.set_ylabel("hit rate")
        axes[0, -1].legend(title="policy")
        fig.suptitle("Eviction policies (exact replay of each device's own trace)")
        fig.tight_layout()
        figs["hit_vs_K"] = fig

    rr = rs.table(None, "real_runs", docs)
    if len(rr):
        streams = list(dict.fromkeys(rr.stream))
        policies = list(dict.fromkeys(rr.policy))
        fig, axes = plt.subplots(len(streams), 2, figsize=(14, 4.5 * len(streams)), squeeze=False)
        for i, stream in enumerate(streams):
            d = rr[rr.stream == stream]
            for ax, col, yl in [(axes[i, 0], "decode_tok_s", "decode tokens / s"),
                                (axes[i, 1], "io_s", "disk-read time (s)")]:
                _grouped_bars(ax, d, "policy", col, policies)
                ax.set_ylabel(yl); ax.set_title(f"{stream}: {yl} at K={int(d.K.iloc[0])}")
        axes[0, 0].legend()
        fig.tight_layout()
        figs["real_cost"] = fig
    return figs


def axis3(docs):
    figs = {}
    r = rs.table(None, "runs", docs)
    if not len(r):
        return figs
    num = ["ttft_s", "GB_loaded", "scratch_GB", "distinct_per_call", "chunk_footprint", "C_num"]
    r = r.assign(C=r.C.astype(str))
    g = r.groupby(["device", "K", "C"], sort=False)[num].mean().reset_index()
    # 'full' (whole prompt in one pass) is drawn one octave right of the largest chunk
    sizes = sorted(int(c) for c in g.C.unique() if c != "full")
    full_x = 2 * sizes[-1]
    g["x"] = [full_x if c == "full" else int(c) for c in g.C]
    ticks = sizes + ([full_x] if "full" in set(g.C) else [])

    def chunk_axis(ax):
        _log2_x(ax, ticks)
        ax.set_xticklabels([str(t) if t != full_x or "full" not in set(g.C) else "full" for t in ticks])
        ax.set_xlabel("chunk size C (tokens)")
    Ks = sorted(g.K.unique())
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, col, title, yl in [(axes[0], "ttft_s", "Time to first token (cold cache)", "s"),
                               (axes[1], "GB_loaded", "Expert bytes read during prefill", "GB"),
                               (axes[2], "scratch_GB", "Scratch memory (peak - bank - weights)", "GB")]:
        for i, K in enumerate(Ks):
            _device_lines(ax, g[g.K == K], "x", col, STYLES[i % 4], MARKERS[i % 4], f" · K={K}")
        chunk_axis(ax)
        ax.axvline(128, color="#999", ls="--", lw=1)
        ax.set_title(title); ax.set_ylabel(yl)
    axes[0].set_yscale("log")
    axes[0].legend()
    fig.suptitle("Prefill chunk size (mean over prompts)")
    fig.tight_layout()
    figs["chunking"] = fig

    d = g.groupby(["device", "x"], sort=False)[["distinct_per_call", "chunk_footprint", "C_num"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    _device_lines(axes[0], d, "x", "distinct_per_call")
    C = np.array(sorted(d.x.unique()), float)
    n_tok = np.array([d[d.x == c].C_num.mean() for c in C])     # tokens per pass ('full' = prompt length)
    axes[0].plot(C, 64 * (1 - (1 - 8 / 64) ** n_tok), color="#999", ls="--", lw=1, label="uniform routing")
    axes[0].axhline(64, color="#999", lw=1)
    axes[0].set_title("Distinct experts per layer call"); axes[0].set_ylabel("experts")
    _device_lines(axes[1], d, "x", "chunk_footprint")
    for K in Ks:
        axes[1].axhline(K, color="#999", ls=":", lw=1)
        axes[1].text(C.max(), K, f" K={K}", fontsize=8, color="#555", va="center")
    axes[1].set_title("Chunk footprint: experts one chunk needs (16 layers)"); axes[1].set_ylabel("experts")
    for ax in axes:
        chunk_axis(ax); ax.legend()
    fig.tight_layout()
    figs["distinct"] = fig
    return figs


def axis4(docs):
    figs = {}
    m = rs.table(None, "micro_reads", docs)
    if len(m):
        mt = m.groupby(["device", "threads", "m"], sort=False).ms.median().reset_index()
        threads = sorted(mt.threads.unique())
        fig, axes = plt.subplots(1, len(threads), figsize=(4.5 * len(threads), 4.2), sharey=True, squeeze=False)
        for ax, T in zip(axes[0], threads):
            _device_lines(ax, mt[mt.threads == T], "m", "ms")
            ax.set_title(f"{T} reader thread{'s' if T > 1 else ''}"); ax.set_xlabel("experts read (m)")
        axes[0, 0].set_ylabel("median ms"); axes[0, 0].legend()
        fig.suptitle("Read latency for m experts (microbenchmark)")
        fig.tight_layout()
        figs["read_latency"] = fig

    c = rs.table(None, "calls_by_m", docs)
    if len(c):
        c = c[c.phase == "decode"]
        modes = list(dict.fromkeys(c["mode"]))
        fig, axes = plt.subplots(1, len(modes), figsize=(4.5 * len(modes), 4.2), sharey=True, squeeze=False)
        for ax, mode in zip(axes[0], modes):
            _device_lines(ax, c[c["mode"] == mode], "m", "t_moe_ms")
            ax.set_title(mode); ax.set_xlabel("misses in the call (m)")
        axes[0, 0].set_ylabel("median MoE-layer ms"); axes[0, 0].legend()
        fig.suptitle("MoE layer time per decode call, by number of misses")
        fig.tight_layout()
        figs["moe_time"] = fig

    r = rs.table(None, "runs", docs)
    if len(r):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        _grouped_bars(axes[0], r, "mode", "decode_tok_s"); axes[0].set_ylabel("tokens / s")
        axes[0].set_title("Decode throughput")
        _grouped_bars(axes[1], r, "mode", "io_s"); axes[1].set_ylabel("s")
        axes[1].set_title("Wall time inside expert reads")
        axes[0].legend()
        fig.tight_layout()
        figs["throughput"] = fig
    return figs


def axis5(docs):
    figs = {}
    rep, run = rs.table(None, "replay", docs), rs.table(None, "runs", docs)
    strategies = list(dict.fromkeys(list(rep.get("strategy", [])) + list(run.get("strategy", []))))
    if not strategies:
        return figs
    fig, axes = plt.subplots(1, len(strategies), figsize=(5 * len(strategies), 4.3), sharey=True, squeeze=False)
    for ax, st in zip(axes[0], strategies):
        if len(rep):
            _device_lines(ax, rep[rep.strategy == st], "B", "hit_rate", "-", "o", " · replay")
        if len(run):
            _device_lines(ax, run[run.strategy == st], "B", "hit_rate", "none", "*", " · timed run", ms=12)
        ax.set_title(st); ax.set_xlabel("expert budget B in fast memory")
    axes[0, 0].set_ylabel("hit rate"); axes[0, 0].legend()
    fig.suptitle("Tiering strategies: hit rate vs budget")
    fig.tight_layout()
    figs["hit_rate"] = fig

    if len(run):
        rows = [("decode_tok_s", "decode tokens / s"), ("ttft_mean_s", "mean TTFT (s)"),
                ("startup_s", "startup preload (s)")]
        fig, axes = plt.subplots(len(rows), len(strategies), figsize=(5 * len(strategies), 3.4 * len(rows)),
                                 sharey="row", squeeze=False)
        for j, st in enumerate(strategies):
            for i, (col, yl) in enumerate(rows):
                ax = axes[i, j]
                _device_lines(ax, run[run.strategy == st], "B", col)
                ax.set_ylabel(yl if j == 0 else "")
                if i == 0:
                    ax.set_title(st)
                if i == len(rows) - 1:
                    ax.set_xlabel("expert budget B in fast memory")
        axes[0, 0].legend()
        fig.suptitle("Tiering strategies: timed runs")
        fig.tight_layout()
        figs["timing"] = fig
    return figs


def axis7(docs):
    figs = {}
    p = rs.table(None, "prediction", docs)
    if len(p):
        p = p[p.phase == "decode"].copy()
        p["rec_n"], p["prec_n"] = p.recall * p.n, p.precision * p.n
        agg = p.groupby(["device", "predictor", "lookahead", "P"], sort=False)[["n", "rec_n", "prec_n"]].sum()
        agg = agg.assign(recall=agg.rec_n / agg.n, precision=agg.prec_n / agg.n).reset_index()
        router = agg[agg.lookahead > 0]
        base = agg[agg.lookahead == 0]
        ds = sorted(router.lookahead.unique())
        fig, axes = plt.subplots(1, len(ds), figsize=(5 * len(ds), 4.3), sharey=True, squeeze=False)
        for ax, d in zip(axes[0], ds):
            _device_lines(ax, router[router.lookahead == d], "P", "recall", "-", "o", " · recall")
            _device_lines(ax, router[router.lookahead == d], "P", "precision", ":", "s", " · precision", alpha=0.7)
            for dev, b in base.groupby("device", sort=False):
                ax.axhline(b.recall.mean(), color=COLORS.get(dev, "grey"), ls="--", lw=1)
            ax.set_title(f"predict layer L+{d} from layer L")
            ax.set_xlabel("experts predicted per token (P)"); ax.set_ylim(0, 1)
        axes[0, 0].set_ylabel("decode accuracy")
        axes[0, -1].legend(loc="lower right", title="dashed line: previous-token recall")
        fig.suptitle("Router-based expert prediction (decode)")
        fig.tight_layout()
        figs["prediction"] = fig

    r = rs.table(None, "runs", docs)
    if len(r):
        modes = list(dict.fromkeys(r["mode"]))
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
        _grouped_bars(axes[0], r, "mode", "decode_tok_s", modes)
        axes[0].set_title("Decode throughput"); axes[0].set_ylabel("tokens / s")
        _grouped_bars(axes[1], r, "mode", "stall_s", modes)
        axes[1].set_title("Stall (demand loads + waiting for prefetch)"); axes[1].set_ylabel("s")
        r2 = r.assign(wasted_share=r.wasted / (r.useful + r.wasted).replace(0, np.nan))
        _grouped_bars(axes[2], r2, "mode", "wasted_share", modes)
        axes[2].set_title("Share of prefetched experts never used"); axes[2].set_ylim(0, 1)
        axes[0].legend()
        fig.tight_layout()
        figs["prefetch_cost"] = fig
    return figs


def axis8(docs):
    figs = {}
    r = rs.table(None, "runs", docs)
    if not len(r):
        return figs
    Ks = sorted(r.K.unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panels = [("agg_tok_s", "Aggregate decode throughput", "tokens / s (all requests)"),
              ("decode_hit_rate", "Decode hit rate", "hit rate"),
              ("loads_per_step", "Expert loads per decode step", "experts / step"),
              ("step_footprint", "Cache lookups per step (Σ layers of union)", "experts / step")]
    for ax, (col, title, yl) in zip(axes.flat, panels):
        for i, K in enumerate(Ks):
            _device_lines(ax, r[r.K == K], "B", col, STYLES[i % 4], MARKERS[i % 4], f" · K={K}")
        _log2_x(ax, r.B)
        ax.set_title(title); ax.set_ylabel(yl); ax.set_xlabel("batch size B")
    B = np.array(sorted(r.B.unique()), float)
    axes[1, 1].plot(B, 16 * np.minimum(64, 8 * B), color="#999", ls="--", lw=1, label="no sharing: 16 x min(64, 8B)")
    for K in Ks:
        axes[1, 1].axhline(K, color="#999", ls=":", lw=1)
        axes[1, 1].text(B.max(), K, f" K={K}", fontsize=8, color="#555", va="center")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 0].legend(); axes[1, 1].legend()
    fig.suptitle("Batch size vs cache contention")
    fig.tight_layout()
    figs["batching"] = fig
    return figs


AXES = {
    "axis-1-cache-capacity": axis1, "axis-2-eviction-policies": axis2, "axis-3-prefill-chunking": axis3,
    "axis-4-io-compute-overlap": axis4, "axis-5-memory-tiering": axis5,
    "axis-7-router-prefetching": axis7, "axis-8-batch-contention": axis8,
}


def make(axis: str, results_dir: Path | str | None = None, out_dir: Path | str | None = None,
         show: bool = True) -> dict:
    """Draw one axis's figures from every device's results; save PNGs if out_dir is given."""
    results_dir = Path(results_dir) if results_dir else EXPERIMENTS / axis / "results"
    docs = rs.load(results_dir)
    if not docs:
        print(f"{axis}: no results in {results_dir}")
        return {}
    with plt.rc_context(STYLE):
        figs = AXES[axis](docs)
        for name, fig in figs.items():
            fig.text(0.995, 0.002, _devices_note(docs), ha="right", va="bottom", fontsize=7, color="#777")
            if out_dir:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                fig.savefig(Path(out_dir) / f"{name}.png", dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            for fig in figs.values():
                plt.close(fig)
    return figs


if __name__ == "__main__":
    for axis in sys.argv[1:] or list(AXES):
        out = EXPERIMENTS / axis / "figures"
        figs = make(axis, out_dir=out, show=False)
        if figs:
            print(f"{axis}: {', '.join(figs)} -> {out.relative_to(EXPERIMENTS.parent)}")
