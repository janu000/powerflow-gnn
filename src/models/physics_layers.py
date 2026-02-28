import jax
import jax.numpy as jnp
import flax.linen as nn

def compute_incidence_matrix(num_nodes: int, num_edges: int, senders: jnp.ndarray, receivers: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the incidence matrix B of shape (num_nodes, num_edges).
    B[i, e] =  1 if edge e enters node i (receivers)
    B[i, e] = -1 if edge e leaves node i (senders)
    """
    B = jnp.zeros((num_nodes, num_edges))
    e_idx = jnp.arange(num_edges)
    B = B.at[receivers, e_idx].set(1.0)
    B = B.at[senders, e_idx].set(-1.0)
    return B

def compute_edge_voltages_kvl(V_node: jnp.ndarray, senders: jnp.ndarray, receivers: jnp.ndarray) -> jnp.ndarray:
    """
    Enforces KVL (Kirchhoff's Voltage Law) by computing edge voltages
    directly from node potentials using sparse gathering.
    
    V_edge = V_node[receivers] - V_node[senders]
    """
    return V_node[receivers] - V_node[senders]

def compute_net_currents(num_nodes: int, I_edge: jnp.ndarray, senders: jnp.ndarray, receivers: jnp.ndarray) -> jnp.ndarray:
    """
    Computes net current entering each node using sparse segment_sum.
    Effectively replaces jnp.dot(B, I_edge).
    """
    # Sum of currents entering the node
    I_in = jax.ops.segment_sum(I_edge, receivers, num_segments=num_nodes)
    # Sum of currents leaving the node
    I_out = jax.ops.segment_sum(I_edge, senders, num_segments=num_nodes)
    return I_in - I_out

def project_currents_kcl(B: jnp.ndarray, I_hat_edge: jnp.ndarray, I_ext: jnp.ndarray) -> jnp.ndarray:
    """
    Projects predicted edge currents to strictly satisfy KCL
    (Kirchhoff's Current Law) using orthogonal projection.
    
    I_edge = I_hat_edge - B^T * pinv(B * B^T) * (B * I_hat_edge - I_ext)
    """
    # B B^T is essentially the Laplacian matrix of the network (N, N)
    L = jnp.dot(B, B.T)
    
    # We use pseudoinverse because L is singular (has at least one zero eigenvalue)
    L_pinv = jnp.linalg.pinv(L)
    
    # Calculate current mismatch at the nodes (N, C)
    # B * I_hat_edge is the net current arriving at each node
    delta_I = jnp.dot(B, I_hat_edge) - I_ext
    
    # Projection term (E, C)
    correction = jnp.dot(B.T, jnp.dot(L_pinv, delta_I))
    
    I_edge = I_hat_edge - correction
    return I_edge

class StrictPhysicsGNNLayer(nn.Module):
    """
    A specialized GNN layer that uses hard architectural enforcement 
    of KVL and KCL based on node potentials and incidence matrices.
    """
    out_dim: int

    @nn.compact
    def __call__(self, V_node: jnp.ndarray, I_ext: jnp.ndarray, senders: jnp.ndarray, receivers: jnp.ndarray, edge_features: jnp.ndarray):
        num_nodes = V_node.shape[0]
        num_edges = senders.shape[0]
        
        # 1. Enforce KVL strictly via sparse gathers
        V_edge = compute_edge_voltages_kvl(V_node, senders, receivers)
        
        # 2. Predict unconstrained currents using an MLP
        edge_inputs = jnp.concatenate([V_edge, edge_features], axis=-1)
        I_hat_edge = nn.Dense(V_node.shape[-1], name='raw_current_mlp')(edge_inputs)
        
        # 3. Enforce KCL strictly (Still uses dense B for pseudoinverse projection)
        B = compute_incidence_matrix(num_nodes, num_edges, senders, receivers)
        I_edge = project_currents_kcl(B, I_hat_edge, I_ext)
        
        # 4. Node Update via sparse scatter
        net_current = compute_net_currents(num_nodes, I_edge, senders, receivers)
        
        node_inputs = jnp.concatenate([V_node, net_current, I_ext], axis=-1)
        V_node_out = nn.Dense(self.out_dim, name='node_update')(node_inputs)
        V_node_out = nn.relu(V_node_out)
        
        return V_node_out, I_edge, V_edge

def compute_complex_current(Y: jnp.ndarray, V_edge: jnp.ndarray) -> jnp.ndarray:
    """
    Computes I = Y * V for complex numbers represented as 2D real arrays [real, imag].
    """
    G, B = Y[..., 0], Y[..., 1]
    V_re, V_im = V_edge[..., 0], V_edge[..., 1]
    I_re = G * V_re - B * V_im
    I_im = G * V_im + B * V_re
    return jnp.stack([I_re, I_im], axis=-1)

class KVLOhmGNNLayer(nn.Module):
    """
    A GNN layer that only enforces KVL and Ohm's Law (Sparse Implementation).
    """
    out_dim: int

    @nn.compact
    def __call__(self, V_node: jnp.ndarray, senders: jnp.ndarray, receivers: jnp.ndarray, edge_features: jnp.ndarray):
        num_nodes = V_node.shape[0]
        
        # 1. Enforce KVL strictly via sparse gathers (Assume first 2 dims are V_re, V_im)
        V_edge = compute_edge_voltages_kvl(V_node[..., :2], senders, receivers)
        
        # 2. Admittance (Y) is given as part of the input graph
        Y_edge = edge_features[..., :2]
        
        # 3. Enforce Ohm's Law strictly using complex multiplication
        I_edge = compute_complex_current(Y_edge, V_edge)
        
        # 4. Node Update via sparse scatter
        net_current = compute_net_currents(num_nodes, I_edge, senders, receivers)
        
        node_inputs = jnp.concatenate([V_node, net_current], axis=-1)
        V_node_out = nn.Dense(self.out_dim, name='node_update')(node_inputs)
        V_node_out = nn.relu(V_node_out)
        
        return V_node_out, I_edge, V_edge

class SoftPhysicsGNNLayer(nn.Module):
    """
    A GNN layer that uses soft physics constraints (Sparse Implementation).
    """
    out_dim: int

    @nn.compact
    def __call__(self, V_node: jnp.ndarray, senders: jnp.ndarray, receivers: jnp.ndarray, edge_features: jnp.ndarray, edge_mask: jnp.ndarray = None):
        num_nodes = V_node.shape[0]
        
        # 1. Physics Path: Compute physical currents via sparse operations
        # Extract the voltage parts (first 2 dims) and admittance
        V_edge_phys = compute_edge_voltages_kvl(V_node[..., :2], senders, receivers)
        Y_edge = edge_features[..., :2]
        I_edge_phys = compute_complex_current(Y_edge, V_edge_phys)
        
        # 2. Neural Path: Predict current residual
        # The network sees the physical voltage, admittance, and the calculated physical current
        # Also include any extra edge features if they exist
        edge_inputs = jnp.concatenate([V_edge_phys, edge_features, I_edge_phys], axis=-1)
        I_edge_residual = nn.Dense(2, name='residual_current_mlp')(edge_inputs)
        
        # Apply mask to silence dummy edges
        if edge_mask is not None:
            I_edge_residual = I_edge_residual * edge_mask[..., None]
        
        # 3. Combine: Final current is physics + learned residual
        I_edge_pred = I_edge_phys + I_edge_residual
        
        # 4. Node Update via sparse scatter
        net_current_pred = compute_net_currents(num_nodes, I_edge_pred, senders, receivers)
        
        node_inputs = jnp.concatenate([V_node, net_current_pred], axis=-1)
        V_node_out = nn.Dense(self.out_dim, name='node_update')(node_inputs)
        V_node_out = nn.relu(V_node_out)
        
        return V_node_out, I_edge_pred
