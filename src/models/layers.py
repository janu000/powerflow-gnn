import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Any, Callable, Optional
from src.data.graph import HyperHeteroMultiGraph, EdgeIndices

class BaseHeteroMessagePassing(nn.Module):
    """
    A modular Heterogeneous Message Passing base layer.
    Users can subclass this and override `message` and `update` to define custom logic
    for specific node or edge types.
    """
    out_dim: int
    aggr: str = 'sum' # 'sum', 'mean', or 'max'

    @nn.compact
    def message(self, edge_type: Tuple[str, str, str], src_feats: jnp.ndarray, dst_feats: jnp.ndarray, edge_feats: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Computes messages on edges. Can be overridden for custom physics/logic.
        """
        src_type, rel_type, dst_type = edge_type
        message_fn = nn.Dense(self.out_dim, name=f'msg_{src_type}_{rel_type}_{dst_type}')
        
        # Base message from sender features
        msgs = message_fn(src_feats)
        
        # Optionally incorporate edge features
        if edge_feats is not None:
            edge_transform = nn.Dense(self.out_dim, name=f'edge_msg_{src_type}_{rel_type}_{dst_type}')
            msgs += edge_transform(edge_feats)
            
        return msgs

    @nn.compact
    def update(self, ntype: str, node_feats: jnp.ndarray, aggr_msgs: jnp.ndarray) -> jnp.ndarray:
        """
        Updates node features given aggregated messages. Can be overridden.
        """
        update_fn = nn.Dense(self.out_dim, name=f'update_{ntype}')
        updated_features = update_fn(jnp.concatenate([node_feats, aggr_msgs], axis=-1))
        return nn.relu(updated_features)

    @nn.compact
    def __call__(self, graph: HyperHeteroMultiGraph, edge_mask: Optional[jnp.ndarray] = None) -> HyperHeteroMultiGraph:
        new_nodes = {ntype: jnp.zeros((features.shape[0], self.out_dim)) 
                     for ntype, features in graph.nodes.items()}
        
        for edge_type, edge_indices in graph.edges.items():
            src_type, rel_type, dst_type = edge_type
            
            # Gather features for senders and receivers
            src_feats_gathered = graph.nodes[src_type][edge_indices.senders]
            dst_feats_gathered = graph.nodes[dst_type][edge_indices.receivers]
            edge_feats = graph.edge_features[edge_type] if graph.edge_features is not None and edge_type in graph.edge_features else None
            
            # 1. Compute Messages
            messages = self.message(edge_type, src_feats_gathered, dst_feats_gathered, edge_feats)
            
            # Apply edge mask if provided
            if edge_mask is not None:
                messages = messages * edge_mask[..., None]
            
            # 2. Aggregate Messages
            num_dst = graph.nodes[dst_type].shape[0]
            receivers = edge_indices.receivers
            
            if self.aggr == 'sum':
                aggr_msgs = jax.ops.segment_sum(messages, receivers, num_segments=num_dst)
            elif self.aggr == 'max':
                aggr_msgs = jax.ops.segment_max(messages, receivers, num_segments=num_dst)
            else: # mean
                aggr_msgs = jax.ops.segment_sum(messages, receivers, num_segments=num_dst)
                counts = jax.ops.segment_sum(jnp.ones_like(messages[..., 0]), receivers, num_segments=num_dst)
                counts = jnp.maximum(counts, 1.0)
                aggr_msgs = aggr_msgs / counts[..., None]
                
            new_nodes[dst_type] += aggr_msgs

        # 3. Update Node Features
        updated_nodes = {}
        for ntype, features in graph.nodes.items():
            updated_nodes[ntype] = self.update(ntype, features, new_nodes[ntype])

        return graph.replace(nodes=updated_nodes)

class HeteroMessagePassing(BaseHeteroMessagePassing):
    """Alias for the standard generic implementation."""
    pass
