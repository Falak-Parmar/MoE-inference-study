# MoE - Inference System

MoE Inference System is a project studying ML inference systems, with a particular focus on Mixture-of-Experts (MoE) models.

The project investigates memory/cache behavior, working sets, and performance characteristics across different hardware configurations.

## Hardware Environments

We target three execution environments:
- Mac with Apple Silicon unified memory + integrated GPU
- Windows with unified/shared memory + integrated GPU
- Windows with dedicated GPU + dedicated VRAM

## Project Structure

- `experiments/` - One folder per research axis (`Mac.ipynb` + `README.md`); see `experiments/README.md`
- `src/moe/` - Shared Python package (`pip install -e ".[mac]"` or `".[windows]"`):
  - `common.py` - Framework-free core: storage index, eviction policies, exact replay, workloads
  - `mlx_cache.py` - Disk-backed K-slot expert cache used by every Mac notebook (MLX)
  - `results.py` - Results file format shared by all three devices (`<axis>/results/<device>.json`)
  - `figures.py` - Cross-device figures (`<axis>/figures/`), one colour per device
- `models/` - Model-specific information and documentation
