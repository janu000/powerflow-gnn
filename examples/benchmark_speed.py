import jax
import jax.numpy as jnp
import flax.linen as nn
import time
from typing import Tuple, Optional
from src.models.physics_layers import KVLOhmGNNLayer, SoftPhysicsGNNLayer
from src.models.layers import BaseHeteroMessagePassing
from src.data.graph import HyperHeteroMultiGraph, EdgeIndices

class StandardGNNLayer(BaseHeteroMessagePassing):
    """An unconstrained baseline GNN layer for comparison."""
    @nn.compact
    def message(self, edge_type: Tuple[str, str, str], src_feats: jnp.ndarray, dst_feats: jnp.ndarray, edge_feats: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        src_type, rel_type, dst_type = edge_type
        
        msg_inputs = [src_feats]
        if edge_feats is not None:
            msg_inputs.append(edge_feats)
        
        message_fn = nn.Dense(self.out_dim, name=f'msg_{src_type}_{rel_type}_{dst_type}')
        return message_fn(jnp.concatenate(msg_inputs, axis=-1))

def generate_graph(key: jax.Array, num_buses: int):
    """Generates a random system."""
    # Let's say ~3 edges per node on average
    num_edges = num_buses * 3
    
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    # Features: [V_re, V_im]
    V_node = jax.random.normal(k1, (num_buses, 2))
    
    # Random edges
    senders = jax.random.randint(k2, (num_edges,), 0, num_buses)
    receivers = jax.random.randint(k3, (num_edges,), 0, num_buses)
    
    # Edge features: Admittance [G, B]
    edge_features = jax.random.normal(k4, (num_edges, 2))
    
    # Build the HyperHeteroMultiGraph representation for the Standard Layer
    nodes = {'bus': V_node}
    edges = {('bus', 'line', 'bus'): EdgeIndices(senders=senders, receivers=receivers)}
    edge_feats_dict = {('bus', 'line', 'bus'): edge_features}
    
    graph = HyperHeteroMultiGraph(
        nodes=nodes, edges=edges, edge_features=edge_feats_dict,
        n_node={'bus': jnp.array([num_buses])},
        n_edge={('bus', 'line', 'bus'): jnp.array([num_edges])}
    )
    
    return graph, V_node, senders, receivers, edge_features

def run_benchmark():
    sizes = [10, 100, 1000, 5000]
    num_passes = 100
    
    print(f"{'Buses':<10} | {'Unconstrained':<18} | {'Soft Physics':<18} | {'Strict (KVL+Ohm)':<18}")
    print("-" * 72)
    
    for num_buses in sizes:
        key = jax.random.PRNGKey(42)
        graph, V_node, senders, receivers, edge_features = generate_graph(key, num_buses)
        
        # 1. Unconstrained GNN
        std_layer = StandardGNNLayer(out_dim=16)
        std_params = std_layer.init(key, graph)
        @jax.jit
        def apply_std(params, g): return std_layer.apply(params, g)
        _ = apply_std(std_params, graph) # Warmup
        start_time = time.time()
        for _ in range(num_passes): _ = apply_std(std_params, graph)
        jax.block_until_ready(_)
        std_time = time.time() - start_time
        
        # 2. Soft Physics GNN Layer
        soft_layer = SoftPhysicsGNNLayer(out_dim=16)
        soft_params = soft_layer.init(key, V_node, senders, receivers, edge_features)
        @jax.jit
        def apply_soft(params, v, s, r, e): return soft_layer.apply(params, v, s, r, e)
        _ = apply_soft(soft_params, V_node, senders, receivers, edge_features) # Warmup
        start_time = time.time()
        for _ in range(num_passes): _ = apply_soft(soft_params, V_node, senders, receivers, edge_features)
        jax.block_until_ready(_)
        soft_time = time.time() - start_time
        
        # 3. Physics-Constrained GNN (KVL + Ohm)
        phys_layer = KVLOhmGNNLayer(out_dim=16)
        phys_params = phys_layer.init(key, V_node, senders, receivers, edge_features)
        @jax.jit
        def apply_phys(params, v, s, r, e): return phys_layer.apply(params, v, s, r, e)
        _ = apply_phys(phys_params, V_node, senders, receivers, edge_features) # Warmup
        start_time = time.time()
        for _ in range(num_passes): _ = apply_phys(phys_params, V_node, senders, receivers, edge_features)
        jax.block_until_ready(_)
        phys_time = time.time() - start_time
        
        print(f"{num_buses:<10} | {std_time:>15.4f} s | {soft_time:>15.4f} s | {phys_time:>15.4f} s")

if __name__ == "__main__":
    run_benchmark()
