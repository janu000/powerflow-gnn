import jax
import jax.numpy as jnp
import flax.linen as nn
from src.data.graph import HyperHeteroMultiGraph, EdgeIndices
from src.models.power_layers import PowerGridMessagePassing

def create_5bus_system() -> HyperHeteroMultiGraph:
    """
    Creates a simple 5-bus network.
    Topology: Bus 0-1, 1-2, 2-3, 3-4, 4-0
    """
    num_buses = 5
    
    # Node features: [V_real, V_imag, P_inj, Q_inj]
    nodes = {
        'bus': jnp.array([
            [1.0, 0.0,  1.5,  0.5],  # Bus 0 (Slack)
            [0.98, -0.1, -0.4, -0.2], # Bus 1 (Load)
            [0.97, -0.15,-0.4, -0.2], # Bus 2 (Load)
            [0.99, -0.05, 0.2,  0.1], # Bus 3 (Gen)
            [0.95, -0.2, -0.5, -0.3], # Bus 4 (Load)
        ])
    }
    
    # Edges: 5 lines forming a ring
    senders = jnp.array([0, 1, 2, 3, 4,  1, 2, 3, 4, 0]) # Bidirectional
    receivers = jnp.array([1, 2, 3, 4, 0,  0, 1, 2, 3, 4])
    
    edges = {
        ('bus', 'line', 'bus'): EdgeIndices(senders=senders, receivers=receivers)
    }
    
    # Edge features: [Conductance (G), Susceptance (B)]
    # Typical values: G is small, B is large negative
    edge_features = {
        ('bus', 'line', 'bus'): jnp.array([
            [1.0, -5.0], [1.2, -4.0], [0.8, -6.0], [1.5, -3.0], [2.0, -2.5],
            [1.0, -5.0], [1.2, -4.0], [0.8, -6.0], [1.5, -3.0], [2.0, -2.5]
        ])
    }
    
    return HyperHeteroMultiGraph(
        nodes=nodes, edges=edges, edge_features=edge_features,
        n_node={'bus': jnp.array([num_buses])},
        n_edge={('bus', 'line', 'bus'): jnp.array([len(senders)])}
    )

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    graph = create_5bus_system()
    
    print("--- Testing Power Grid GNN ---")
    
    # 1. Baseline: Pure Neural Network (No Physics Bias)
    print("\n[1] Standard GNN (No Physics Bias)")
    layer_standard = PowerGridMessagePassing(out_dim=16, use_physics_bias=False)
    params_std = layer_standard.init(key, graph)
    out_std = layer_standard.apply(params_std, graph)
    print(f"Output bus feature shape: {out_std.nodes['bus'].shape}")
    
    # 2. Soft Physics Bias: MLP sees calculated Current
    print("\n[2] Soft Physics Bias (Injects Physical Current as Feature)")
    layer_soft = PowerGridMessagePassing(out_dim=16, use_physics_bias=True, bias_type='soft')
    params_soft = layer_soft.init(key, graph)
    out_soft = layer_soft.apply(params_soft, graph)
    print("Successfully ran Soft Bias.")
    
    # 3. Hard Physics Bias: Forces output to be Residual of Current
    print("\n[3] Hard Physics Bias (Residual Connection of Current)")
    layer_hard = PowerGridMessagePassing(out_dim=16, use_physics_bias=True, bias_type='hard')
    params_hard = layer_hard.init(key, graph)
    out_hard = layer_hard.apply(params_hard, graph)
    print("Successfully ran Hard Bias.")
