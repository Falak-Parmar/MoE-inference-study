# MoE - Inference System

MoE Inference System is a project studying ML inference systems, with a particular focus on Mixture-of-Experts (MoE) models.

The project investigates memory/cache behavior, working sets, and performance characteristics across different hardware configurations.

## Hardware Environments

We target three execution environments:
- Mac with Apple Silicon unified memory + integrated GPU
- Windows with unified/shared memory + integrated GPU
- Windows with dedicated GPU + dedicated VRAM

## Project Structure

- `experiments/` - Experimental axes and their implementations
- `models/` - Model-specific information and documentation
- `src/` - Source code and utilities (upcoming)
- `inputs/` - Input prompts (upcoming)
