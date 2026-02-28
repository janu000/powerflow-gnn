# Hyper Heterogeneous Multi-Graph Neural Network Framework

This repository provides a highly modular, JAX/Flax-based framework for training Graph Neural Networks (GNNs) on complex graph structures, with a specialized focus on **Physics-Informed Power Grid Modeling**.

## Features

- **Physics-Informed Architectures**: Native layers that strictly enforce Kirchhoff's Voltage Law (KVL) and Ohm's Law directly within the neural architecture using implicit sparse tensor operations.
- **Global Super-Node Attention**: Custom $O(1)$ global routing layers that allow massive grid structures to communicate across long electrical distances instantly.
- **Heterogeneous Graphs**: Supports multiple node types (generators, loads) and complex edge types (lines, transformers).
- **High Performance**: Built natively on JAX and Flax for lightning-fast JIT compilation, leveraging static graph padding to eliminate recompilation bottlenecks across varying topologies.

## Library Structure

The framework is designed as a reusable library:
- `src.data.graph`: The core `HyperHeteroMultiGraph` PyTree structure.
- `src.models.physics_layers`: Contains `StrictPhysicsGNNLayer` and `SoftPhysicsGNNLayer`.
- `src.models.attention_layers`: Contains `HeteroGATLayer` and `GlobalSuperNodeLayer`.
- `src.models.power_flow_models`: Ready-to-use AC power flow solving architectures.

## Setup

Install dependencies using `uv` (includes `jax`, `flax`, `pandapower`, and `numba`):

```bash
uv sync
```

## Workflows

### 1. Data Generation
Generate a dataset of variable-topology AC power grids (solved via Newton-Raphson):

```bash
uv run python src/data/generate_100bus_dataset.py
```
*(This uses `pandapower` to generate 200 grids with ~100 buses each, saving the ground-truth voltages to `data/power_dataset_100bus.pkl`)*

### 2. Training and Benchmarking
Train and benchmark various architectures (Unconstrained, Soft Physics, Strict Physics, and Global Super-Node variants) against the generated dataset. 

```bash
export PYTHONPATH=$PYTHONPATH:.
uv run python src/training/train_supernode.py
```
*(Results are automatically appended to `training_results.md`)*

### 3. Running Unit Tests
Validate the strict mathematical guarantees of the physics layers:

```bash
export PYTHONPATH=$PYTHONPATH:.
uv run pytest tests/
```