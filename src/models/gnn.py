import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence
from src.data.graph import HyperHeteroMultiGraph
from src.models.layers import HeteroMessagePassing

class HyperHeteroGNN(nn.Module):
    """
    Main GNN Model for Hyper Heterogeneous Multi-Graphs.
    """
    hidden_dims: Sequence[int]
    out_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, graph: HyperHeteroMultiGraph) -> jnp.ndarray:
        # 1. Apply multiple rounds of message passing
        for i in range(self.num_layers):
            graph = HeteroMessagePassing(
                out_dim=self.hidden_dims[i] if i < len(self.hidden_dims) else self.hidden_dims[-1],
                aggr='mean',
                name=f'hetero_mp_layer_{i}'
            )(graph)

        # 2. Readout / Global pooling (if global prediction is needed)
        # Here we perform an example: sum pooling across all nodes of a specific target type
        # Or simply pool everything into a single representation.
        # This part heavily depends on the downstream task (node classification vs graph classification).
        # We assume Graph Classification for this template.
        
        pooled_reprs = []
        for ntype, features in graph.nodes.items():
            pooled = jnp.sum(features, axis=0)
            pooled_reprs.append(pooled)
            
        global_repr = jnp.concatenate(pooled_reprs, axis=-1)
        
        # 3. Final MLP for classification / regression
        out = nn.Dense(self.hidden_dims[-1], name='mlp_hidden')(global_repr)
        out = nn.relu(out)
        out = nn.Dense(self.out_dim, name='mlp_out')(out)
        
        return out
