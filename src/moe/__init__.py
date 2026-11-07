"""Shared code for the MoE inference study (OLMoE expert caching on three devices).

    moe.common      framework-free core: storage index, eviction policies, exact replay,
                    recorder, workloads, analysis (used by every device)
    moe.results     results file format: experiments/<axis>/results/<device>.json
    moe.figures     cross-device figures (python -m moe.figures)
    moe.mlx_cache   Mac / MLX backend: the true K-slot expert cache (needs mlx, mlx-lm)

Submodules are imported explicitly (``from moe import mlx_cache as mc``) so that
importing ``moe`` never pulls in MLX or PyTorch on a machine that lacks them.
"""
