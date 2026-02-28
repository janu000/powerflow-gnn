import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional, Dict
from src.data.graph import HyperHeteroMultiGraph

def segment_softmax(data: jnp.ndarray, segment_ids: jnp.ndarray, num_segments: int) -> jnp.ndarray:
    """Computes a numerically stable softmax over segments."""
    # max for numerical stability
    max_data = jax.ops.segment_max(data, segment_ids, num_segments=num_segments)
    data_minus_max = data - max_data[segment_ids]
    
    exp_data = jnp.exp(data_minus_max)
    sum_exp = jax.ops.segment_sum(exp_data, segment_ids, num_segments=num_segments)
    
    return exp_data / (sum_exp[segment_ids] + 1e-8)

class HeteroGATLayer(nn.Module):
    """
    Heterogeneous Graph Attention Layer.
    Computes local attention over explicit physical edges.
    """
    out_dim: int
    num_heads: int = 1

    @nn.compact
    def __call__(self, graph: HyperHeteroMultiGraph) -> HyperHeteroMultiGraph:
        head_dim = self.out_dim // self.num_heads
        
        # 1. Project node features
        proj_nodes = {}
        for ntype, feats in graph.nodes.items():
            proj_nodes[ntype] = nn.Dense(self.out_dim, name=f'gat_proj_{ntype}')(feats)
            # Reshape for multi-head attention: (N, num_heads, head_dim)
            proj_nodes[ntype] = proj_nodes[ntype].reshape((feats.shape[0], self.num_heads, head_dim))
            
        new_nodes = {ntype: jnp.zeros_like(proj_nodes[ntype]) for ntype in graph.nodes.items()}
        
        # 2. Compute Attention and Messages per edge type
        for edge_type, edge_indices in graph.edges.items():
            src_type, rel_type, dst_type = edge_type
            senders, receivers = edge_indices.senders, edge_indices.receivers
            num_dst = graph.nodes[dst_type].shape[0]
            
            src_feats = proj_nodes[src_type][senders]
            dst_feats = proj_nodes[dst_type][receivers]
            
            # Incorporate edge features if available
            edge_feats_proj = 0.0
            if graph.edge_features is not None and edge_type in graph.edge_features:
                e_feats = graph.edge_features[edge_type]
                e_proj = nn.Dense(self.out_dim, name=f'edge_proj_{rel_type}')(e_feats)
                edge_feats_proj = e_proj.reshape((e_feats.shape[0], self.num_heads, head_dim))
            
            # Concatenate [src || dst || edge]
            concat_feats = jnp.concatenate([src_feats, dst_feats, src_feats + edge_feats_proj], axis=-1)
            
            # Attention coefficients (E, num_heads, 1)
            attn_src = nn.Dense(1, use_bias=False, name=f'attn_{rel_type}')(concat_feats)
            attn_src = nn.leaky_relu(attn_src, negative_slope=0.2)
            
            # Softmax over receivers to get alpha
            alpha = jax.vmap(segment_softmax, in_axes=(1, None, None), out_axes=1)(
                attn_src, receivers, num_dst
            )
            
            # Weight messages and aggregate
            messages = src_feats * alpha
            
            aggr_msgs = jax.vmap(jax.ops.segment_sum, in_axes=(1, None, None), out_axes=1)(
                messages, receivers, num_dst
            )
            
            new_nodes[dst_type] += aggr_msgs

        # 3. Update Node Features (concatenate heads)
        updated_nodes = {}
        for ntype, feats in proj_nodes.items():
            # Concat original projected features with aggregated messages
            node_input = jnp.concatenate([feats, new_nodes[ntype]], axis=-1)
            # Flatten heads
            node_input_flat = node_input.reshape((node_input.shape[0], -1))
            
            updated = nn.Dense(self.out_dim, name=f'update_{ntype}')(node_input_flat)
            updated_nodes[ntype] = nn.relu(updated)

        return graph.replace(nodes=updated_nodes)

class GlobalSuperNodeLayer(nn.Module):
    """
    Provides global receptive field in O(1) hops by utilizing a global latent vector.
    Nodes -> Attention -> Global State -> Broadcast -> Nodes.
    """
    out_dim: int

    @nn.compact
    def __call__(self, graph: HyperHeteroMultiGraph) -> HyperHeteroMultiGraph:
        # Initialize or fetch global state
        if graph.globals is None:
            global_state = jnp.zeros((1, self.out_dim))
        else:
            global_state = graph.globals
            
        # 1. READOUT: Nodes attend to Global State
        pooled_messages = []
        for ntype, feats in graph.nodes.items():
            # Project nodes
            proj_feats = nn.Dense(self.out_dim, name=f'global_proj_{ntype}')(feats)
            
            # Compute attention score: how much does the global state care about this node?
            # Dot product attention between node feats and global state
            attn_scores = jnp.dot(proj_feats, global_state.T) # (N, 1)
            alpha = jax.nn.softmax(attn_scores, axis=0) # Softmax over all nodes of this type
            
            # Weighted sum
            pool = jnp.sum(proj_feats * alpha, axis=0, keepdims=True)
            pooled_messages.append(pool)
            
        # Update global state
        all_pooled = jnp.sum(jnp.stack(pooled_messages, axis=0), axis=0)
        new_global_state = nn.Dense(self.out_dim, name='global_update')(
            jnp.concatenate([global_state, all_pooled], axis=-1)
        )
        new_global_state = nn.relu(new_global_state)
        
        # 2. BROADCAST: Global State updates Nodes
        updated_nodes = {}
        for ntype, feats in graph.nodes.items():
            # Broadcast the new global state to all nodes of this type
            broadcasted_global = jnp.tile(new_global_state, (feats.shape[0], 1))
            
            # Update node incorporating global information
            updated = nn.Dense(self.out_dim, name=f'node_global_update_{ntype}')(
                jnp.concatenate([feats, broadcasted_global], axis=-1)
            )
            updated_nodes[ntype] = nn.relu(updated)

        return graph.replace(nodes=updated_nodes, globals=new_global_state)
