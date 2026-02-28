import jax.numpy as jnp
from typing import Dict
from src.models.physics_layers import compute_edge_voltages_kvl, compute_complex_current, compute_net_currents

def mse_loss(preds, targets, mask):
    loss = ((preds - targets) ** 2) * mask[..., None]
    return jnp.sum(loss) / (jnp.sum(mask) * preds.shape[-1] + 1e-8)

def calculate_metrics(V_pred: jnp.ndarray, batch: dict) -> Dict[str, jnp.ndarray]:
    """
    Calculates advanced power flow metrics.
    V_pred: (N, 2) [V_real, V_imag]
    """
    V_true = batch['labels']
    nodes = batch['nodes'] 
    senders = batch['edges']['senders']
    receivers = batch['edges']['receivers']
    Y_edge = batch['edge_features'][..., :2]
    node_mask = batch['node_mask']
    num_nodes = V_pred.shape[0]

    # 1. Voltage Accuracy (MSE on Magnitude)
    V_mag_pred = jnp.sqrt(V_pred[:, 0]**2 + V_pred[:, 1]**2 + 1e-8)
    V_mag_true = jnp.sqrt(V_true[:, 0]**2 + V_true[:, 1]**2 + 1e-8)
    vol_acc = jnp.sum(((V_mag_pred - V_mag_true)**2) * node_mask) / (jnp.sum(node_mask) + 1e-8)

    # 2. Power Mismatch
    # Calculate physical current I = Y * \Delta V
    V_edge = compute_edge_voltages_kvl(V_pred, senders, receivers)
    I_edge = compute_complex_current(Y_edge, V_edge)
    
    # Net current entering nodes
    I_net = compute_net_currents(num_nodes, I_edge, senders, receivers)
    
    # Complex Power S = V * I^* 
    V_re, V_im = V_pred[:, 0], V_pred[:, 1]
    I_re, I_im = I_net[:, 0], I_net[:, 1]
    
    P_calc = V_re * I_re + V_im * I_im
    Q_calc = V_im * I_re - V_re * I_im
    
    # Ground truth injections
    P_inj, Q_inj = nodes[:, 0], nodes[:, 1]
    
    # Mismatch mask: ignore padded nodes AND ignore the slack bus (index 0)
    mismatch_mask = jnp.copy(node_mask)
    mismatch_mask = mismatch_mask.at[0].set(False)
    num_mismatch_nodes = jnp.sum(mismatch_mask) + 1e-8
    
    P_mismatch = jnp.sum(jnp.abs(P_calc - P_inj) * mismatch_mask) / num_mismatch_nodes
    Q_mismatch = jnp.sum(jnp.abs(Q_calc - Q_inj) * mismatch_mask) / num_mismatch_nodes
    
    # 3. Feasibility Rate
    max_p_mismatch = jnp.max(jnp.abs(P_calc - P_inj) * mismatch_mask)
    max_q_mismatch = jnp.max(jnp.abs(Q_calc - Q_inj) * mismatch_mask)
    max_mismatch = jnp.maximum(max_p_mismatch, max_q_mismatch)
    feasible = jnp.where(max_mismatch < 1e-2, 1.0, 0.0) # 0.01 pu tolerance ~ 1 MVA
    
    return {
        'mse': mse_loss(V_pred, V_true, node_mask),
        'vol_acc': vol_acc,
        'p_mismatch': P_mismatch,
        'q_mismatch': Q_mismatch,
        'feasible': feasible
    }
