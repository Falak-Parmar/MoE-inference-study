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
- `experiments/moe_cache.py` - Shared disk-backed expert cache used by every Mac notebook (MLX)
- `experiments/results.py` - Results file format shared by all three devices (`<axis>/results/<device>.json`)
- `experiments/figures.py` - Cross-device figures (`<axis>/figures/`), one colour per device
- `models/` - Model-specific information and documentation
