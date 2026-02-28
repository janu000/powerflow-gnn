import jax
import jax.numpy as jnp
import chex
from src.models.physics_layers import StrictPhysicsGNNLayer, KVLOhmGNNLayer, SoftPhysicsGNNLayer, compute_incidence_matrix

def test_strict_physics_gnn_layer():
    key = jax.random.PRNGKey(42)
    
    # Simple graph: 3 nodes, 2 edges
    # 0 -> 1, 1 -> 2
    num_nodes = 3
    num_edges = 2
    senders = jnp.array([0, 1])
    receivers = jnp.array([1, 2])
    
    # Dummy features
    V_node = jnp.array([[1.0, 0.0], [0.8, -0.1], [0.6, -0.2]]) # (3, 2)
    I_ext = jnp.array([[0.5, 0.1], [0.0, 0.0], [-0.5, -0.1]])  # (3, 2)
    edge_features = jnp.array([[1.0, -5.0], [1.2, -4.0]]) # (2, 2)
    
    layer = StrictPhysicsGNNLayer(out_dim=4)
    variables = layer.init(key, V_node, I_ext, senders, receivers, edge_features)
    
    V_node_out, I_edge, V_edge = layer.apply(variables, V_node, I_ext, senders, receivers, edge_features)
    
    # 1. Check KVL
    B = compute_incidence_matrix(num_nodes, num_edges, senders, receivers)
    V_edge_expected = jnp.dot(B.T, V_node)
    chex.assert_trees_all_close(V_edge, V_edge_expected)
    
    # 2. Check KCL
    # The net current at each node should be exactly equal to I_ext
    # Because B * I_edge = I_ext
    I_ext_calc = jnp.dot(B, I_edge)
    chex.assert_trees_all_close(I_ext_calc, I_ext, atol=1e-5, rtol=1e-5)

    # 3. Check output shape
    chex.assert_shape(V_node_out, (3, 4))

def test_kvl_ohm_gnn_layer():
    key = jax.random.PRNGKey(123)
    
    num_nodes = 3
    num_edges = 2
    senders = jnp.array([0, 1])
    receivers = jnp.array([1, 2])
    
    V_node = jnp.array([[1.0, 0.0], [0.8, -0.1], [0.6, -0.2]]) # (3, 2)
    edge_features = jnp.array([[1.0, -5.0], [1.2, -4.0]]) # (2, 2)
    
    layer = KVLOhmGNNLayer(out_dim=4)
    variables = layer.init(key, V_node, senders, receivers, edge_features)
    
    V_node_out, I_edge, V_edge = layer.apply(variables, V_node, senders, receivers, edge_features)
    
    # 1. Check KVL
    B = compute_incidence_matrix(num_nodes, num_edges, senders, receivers)
    V_edge_expected = jnp.dot(B.T, V_node)
    chex.assert_trees_all_close(V_edge, V_edge_expected)
    
    # 2. Check output shapes
    chex.assert_shape(V_node_out, (3, 4))
    chex.assert_shape(I_edge, (2, 2)) # 2 edges, 2 feature dimensions

def test_soft_physics_gnn_layer():
    key = jax.random.PRNGKey(456)
    
    senders = jnp.array([0, 1])
    receivers = jnp.array([1, 2])
    
    V_node = jnp.array([[1.0, 0.0], [0.8, -0.1], [0.6, -0.2]]) # (3, 2)
    edge_features = jnp.array([[1.0, -5.0], [1.2, -4.0]]) # (2, 2)
    
    layer = SoftPhysicsGNNLayer(out_dim=4)
    variables = layer.init(key, V_node, senders, receivers, edge_features)
    
    V_node_out, I_edge_pred = layer.apply(variables, V_node, senders, receivers, edge_features)
    
    chex.assert_shape(V_node_out, (3, 4))
    chex.assert_shape(I_edge_pred, (2, 2))

