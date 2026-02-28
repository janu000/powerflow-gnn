# Hyper Heterogeneous Multi-Graph Neural Network Framework

A modular, JAX/Flax-powered framework for training Graph Neural Networks (GNNs) on highly complex graph structures, including Hypergraphs, Heterogeneous graphs, and Multi-Graphs.

## Project Overview

This framework is designed for high-performance GNN research and development. It leverages JAX for JIT compilation and hardware acceleration (CPU/GPU/TPU) and Flax for flexible neural network definitions.

### Key Concepts
- **Hypergraphs**: Modeled using a bipartite graph approach where hyperedges are treated as specific node types.
- **Heterogeneous Graphs**: Supports multiple node and relation types with type-specific message passing.
- **Multi-Graphs**: Allows multiple edges between nodes, handled through distinct relation types or edge indices.

## Core Architecture

- **`src/data/graph.py`**: Defines `HyperHeteroMultiGraph`, a Flax-compatible PyTree (`struct.dataclass`) for representing complex graph structures.
- **`src/models/layers.py`**: Implements `HeteroMessagePassing`, the core layer that handles message generation, aggregation, and node updates across different relation types.
- **`src/models/gnn.py`**: The top-level `HyperHeteroGNN` model that stacks message-passing layers and performs graph-level readout.
- **`src/training/trainer.py`**: Contains JAX-jitted `train_step` and `eval_step` functions for efficient optimization using Optax.
- **`src/data/dataset.py`**: Provides `DummyDataset` and `generate_dummy_graph` for prototyping and testing the framework.

## Getting Started

### Prerequisites
- Python >= 3.14
- `uv` (Fast Python package manager)

### Installation
Sync the environment and install dependencies:
```bash
uv sync
```

### Running Training
Run the main training loop with a configuration file:
```bash
uv run main.py --config configs/default.yaml
```

### Running Tests
Run the unit tests using `pytest`:
```bash
export PYTHONPATH=$PYTHONPATH:.
uv run pytest
```

## Development Conventions

- **JAX/Flax Integration**: Follow JAX best practices (pure functions, immutability). Use `jax.jit` for performance-critical functions and `nn.compact` for Flax modules.
- **Data Structures**: Use the `HyperHeteroMultiGraph` dataclass for all graph-related operations to ensure compatibility with JAX transformations (jit, grad, vmap).
- **Modularity**: Keep data processing (`src/data`), model definitions (`src/models`), and training logic (`src/training`) decoupled.
- **Configuration**: Use YAML files in `configs/` to manage hyperparameters and model settings.

## TODO / Future Work
- Implement more advanced readout strategies (e.g., attention-based pooling).
- Add support for edge feature updates during message passing.
- Implement comprehensive unit tests using `chex` and `pytest`.
- Add real-world dataset loaders (e.g., OGB, IMDB, DBLP).
