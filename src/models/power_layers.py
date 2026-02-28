import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional
from src.models.layers import BaseHeteroMessagePassing

class PowerGridMessagePassing(BaseHeteroMessagePassing):
    """
    A specific message passing layer for Power Grid modeling.
    Extends the BaseHeteroMessagePassing to inject Ohm's Law and KCL priors.
    """
    use_physics_bias: bool = False
    bias_type: str = 'soft'  # 'soft' or 'hard'

    @nn.compact
    def message(self, edge_type: Tuple[str, str, str], src_feats: jnp.ndarray, dst_feats: jnp.ndarray, edge_feats: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        src_type, rel_type, dst_type = edge_type
        
        msg_inputs = [src_feats]
        
        # --- MODULAR PHYSICS INJECTION ---
        I_phys = None
        if self.use_physics_bias and rel_type == 'line' and edge_feats is not None:
            # Assume Node features [0:2] are (V_real, V_imag)
            v_src_re, v_src_im = src_feats[:, 0], src_feats[:, 1]
            v_dst_re, v_dst_im = dst_feats[:, 0], dst_feats[:, 1]
            
            # Assume Edge features [0:2] are Admittance (G, B)
            G, B = edge_feats[:, 0], edge_feats[:, 1]
            
            # Ohm's Law: I = Y * (V_src - V_dst)
            delta_v_re = v_src_re - v_dst_re
            delta_v_im = v_src_im - v_dst_im
            
            I_re = G * delta_v_re - B * delta_v_im
            I_im = G * delta_v_im + B * delta_v_re
            I_phys = jnp.stack([I_re, I_im], axis=-1)  # Shape: (E, 2)
            
            if self.bias_type == 'soft':
                # Soft constraint: Let the MLP "see" the exact physical current
                msg_inputs.append(I_phys)
                
        if edge_feats is not None:
            msg_inputs.append(edge_feats)
        
        # Neural Message Function
        message_fn = nn.Dense(self.out_dim, name=f'msg_{src_type}_{rel_type}_{dst_type}')
        messages = message_fn(jnp.concatenate(msg_inputs, axis=-1))
        
        # Hard structural constraint
        if self.use_physics_bias and rel_type == 'line' and self.bias_type == 'hard' and I_phys is not None:
            # Project the 2D current to the output dimension to add as a residual
            phys_proj = nn.Dense(self.out_dim, use_bias=False, name='phys_proj')(I_phys)
            messages = messages + phys_proj
            
        return messages
