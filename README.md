# Hyper Heterogeneous Multi-Graph Neural Network Framework (Experimental)

This repository contains an experimental, JAX/Flax-based Graph Neural Network (GNN) framework. Its primary focus is exploring Physics-Informed Power Grid Modeling on complex graph structures. 

**Note: This is a research prototype, not a production-ready solver.** It is designed to test the effects of architectural inductive biases (like Kirchhoff's laws) versus purely unconstrained neural approaches.

## Core Concepts

- **Physics-Informed Layers**: Includes experimental layers (`StrictPhysicsGNNLayer`, `SoftPhysicsGNNLayer`) that attempt to embed Kirchhoff's Voltage Law (KVL) and Ohm's Law directly into the message-passing step.
- **Global Super-Node Attention**: A mechanism to allow $O(1)$ global routing across the graph to mitigate the limited receptive field of standard GNNs.
- **Heterogeneous Graphs**: Uses a custom `HyperHeteroMultiGraph` PyTree to support multiple node and edge types natively in JAX.

## Known Limitations

- **AC Power Flow Constraints**: The current "strict" physics model guarantees intermediate physical properties ($I = Y \cdot \Delta V$) but does *not* natively guarantee that the final $P_{calc}$ matches the $P_{injected}$ boundary conditions without a physics-informed loss function (PINN).
- **Scale**: While the data loader uses static padding to speed up JAX compilation, the models have only been validated on small-to-medium synthetic networks (~100 buses).
- **Solver Feasibility**: The neural networks currently output approximations (low MSE) but fail to achieve 100% strict feasibility tolerances ($10^{-6}$) required by real-world utilities compared to standard Newton-Raphson solvers.

## Library Structure

- `src.data.graph`: The `HyperHeteroMultiGraph` data structure.
- `src.models.physics_layers`: Physics-constrained message passing layers.
- `src.models.attention_layers`: Attention and Global Super-Node layers.
- `src.models.power_flow_models`: Composed GNN architectures for AC power flow estimation.

## Setup

Install dependencies using `uv` (includes `jax`, `flax`, `pandapower`, and `numba`):

```bash
uv sync
```

## Workflows

### 1. Data Generation
Generate a synthetic dataset of variable-topology AC power grids (solved via Newton-Raphson):

```bash
uv run python src/data/generate_100bus_dataset.py
```
*(Uses `pandapower` to generate 200 random grids with ~100 buses each, saving to `data/power_dataset_100bus.pkl`)*

### 2. Training and Benchmarking
Train and compare the Unconstrained, Soft Physics, and Strict Physics models (with and without Global Super-Nodes):

```bash
export PYTHONPATH=$PYTHONPATH:.
uv run python src/training/train_supernode.py
```

### 3. Running Unit Tests
Validate the mathematical constraints of the physics layers:

```bash
export PYTHONPATH=$PYTHONPATH:.
uv run pytest tests/
```