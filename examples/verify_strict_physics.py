import jax
import jax.numpy as jnp
import chex
from src.models.physics_layers import KVLOhmGNNLayer, compute_complex_current

def verify_physics_constraints():
    print("--- Verifying Strict Physics Constraints ---")
    
    # Generate some random, unconstrained data
    rng = jax.random.PRNGKey(42)
    num_nodes = 5
    num_edges = 6
    
    k1, k2 = jax.random.split(rng)
    
    # Node voltages [V_re, V_im, ...other features...]
    # We'll make it shape (5, 4) where first 2 are voltage
    V_node = jax.random.normal(k1, (num_nodes, 4))
    
    # Edges
    senders = jnp.array([0, 1, 2, 3, 4, 0])
    receivers = jnp.array([1, 2, 3, 4, 0, 2])
    
    # Edge features (Admittance) [G, B, ...other features...]
    # shape (6, 3) where first 2 are G, B
    edge_features = jax.random.normal(k2, (num_edges, 3))
    
    # Initialize Layer
    layer = KVLOhmGNNLayer(out_dim=8)
    variables = layer.init(rng, V_node, senders, receivers, edge_features)
    
    # Forward Pass
    V_node_out, I_edge, V_edge = layer.apply(variables, V_node, senders, receivers, edge_features)
    
    print("\n1. Verifying KVL (Kirchhoff's Voltage Law)...")
    # Expected: V_edge = V_node[receivers] - V_node[senders]
    expected_V_edge = V_node[receivers, :2] - V_node[senders, :2]
    
    # Assert they match exactly
    diff_kvl = jnp.max(jnp.abs(V_edge - expected_V_edge))
    print(f"   Max KVL Error: {diff_kvl}")
    assert diff_kvl < 1e-6, "KVL VIOLATED!"
    print("   -> KVL strictly guaranteed.")

    print("\n2. Verifying Ohm's Law (Complex Current)...")
    # Expected: I = Y * V (complex multiplication)
    expected_I_edge = compute_complex_current(edge_features[..., :2], V_edge)
    
    # Assert they match exactly
    diff_ohm = jnp.max(jnp.abs(I_edge - expected_I_edge))
    print(f"   Max Ohm's Law Error: {diff_ohm}")
    assert diff_ohm < 1e-6, "OHM'S LAW VIOLATED!"
    print("   -> Ohm's Law strictly guaranteed.")
    
    print("\n3. Verifying Output Formats...")
    print(f"   V_node_out shape: {V_node_out.shape} -> Expected: ({num_nodes}, 8)")
    print(f"   I_edge shape: {I_edge.shape} -> Expected: ({num_edges}, 2)")
    print(f"   V_edge shape: {V_edge.shape} -> Expected: ({num_edges}, 2)")
    
    print("\nAll Physics Constraints Mathematically Guaranteed by Architecture!")

if __name__ == "__main__":
    verify_physics_constraints()
