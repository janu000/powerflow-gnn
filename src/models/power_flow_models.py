import jax.numpy as jnp
import flax.linen as nn
from src.models.physics_layers import SoftPhysicsGNNLayer, KVLOhmGNNLayer
from src.models.layers import BaseHeteroMessagePassing
from src.models.attention_layers import GlobalSuperNodeLayer
from src.data.graph import HyperHeteroMultiGraph, EdgeIndices

class UnconstrainedLayer(BaseHeteroMessagePassing):
    @nn.compact
    def message(self, edge_type, src_feats, dst_feats, edge_feats=None):
        msg_inputs = [src_feats]
        if edge_feats is not None:
            msg_inputs.append(edge_feats)
        return nn.Dense(self.out_dim)(jnp.concatenate(msg_inputs, axis=-1))

# --- Local 3-Layer GNNs ---

class PowerFlowSoftGNN(nn.Module):
    hidden_dim: int = 32
    num_layers: int = 3
    
    @nn.compact
    def __call__(self, P_Q_inj, senders, receivers, edge_features, edge_mask=None):
        V_pred = jnp.zeros_like(P_Q_inj)
        V_pred = V_pred.at[:, 0].set(1.0)
        h = nn.Dense(self.hidden_dim)(P_Q_inj)
        
        for _ in range(self.num_layers):
            v_input = jnp.concatenate([V_pred, h], axis=-1)
            v_out, _ = SoftPhysicsGNNLayer(out_dim=self.hidden_dim)(v_input, senders, receivers, edge_features, edge_mask)
            h = nn.relu(v_out)
            delta_V = nn.Dense(2)(h)
            V_pred = V_pred + delta_V
            
        return V_pred

class PowerFlowStrictGNN(nn.Module):
    hidden_dim: int = 32
    num_layers: int = 3
    
    @nn.compact
    def __call__(self, P_Q_inj, senders, receivers, edge_features, edge_mask=None):
        V_pred = jnp.zeros_like(P_Q_inj)
        V_pred = V_pred.at[:, 0].set(1.0)
        h = nn.Dense(self.hidden_dim)(P_Q_inj)
        
        for _ in range(self.num_layers):
            v_input = jnp.concatenate([V_pred, h], axis=-1)
            v_out, _, _ = KVLOhmGNNLayer(out_dim=self.hidden_dim)(v_input, senders, receivers, edge_features) 
            h = nn.relu(v_out)
            delta_V = nn.Dense(2)(h)
            V_pred = V_pred + delta_V
            
        return V_pred

class PowerFlowUnconstrainedGNN(nn.Module):
    hidden_dim: int = 32
    num_layers: int = 3
    
    @nn.compact
    def __call__(self, P_Q_inj, senders, receivers, edge_features, edge_mask=None):
        V_pred = jnp.zeros_like(P_Q_inj)
        V_pred = V_pred.at[:, 0].set(1.0)
        h = nn.Dense(self.hidden_dim)(P_Q_inj)
        
        for _ in range(self.num_layers):
            node_inputs = jnp.concatenate([V_pred, h], axis=-1)
            layer = UnconstrainedLayer(out_dim=self.hidden_dim)
            dummy_graph = HyperHeteroMultiGraph(
                nodes={'bus': node_inputs},
                edges={('bus', 'line', 'bus'): EdgeIndices(senders=senders, receivers=receivers)},
                edge_features={('bus', 'line', 'bus'): edge_features}
            )
            out_graph = layer(dummy_graph, edge_mask=edge_mask)
            h = nn.relu(out_graph.nodes['bus'])
            delta_V = nn.Dense(2)(h)
            V_pred = V_pred + delta_V
            
        return V_pred

# --- Global Super-Node GNNs ---

class PowerFlowSoftSuperNodeGNN(nn.Module):
    hidden_dim: int = 32
    num_layers: int = 3
    
    @nn.compact
    def __call__(self, P_Q_inj, senders, receivers, edge_features, edge_mask=None):
        V_pred = jnp.zeros_like(P_Q_inj)
        V_pred = V_pred.at[:, 0].set(1.0)
        h = nn.Dense(self.hidden_dim)(P_Q_inj)
        
        global_state = jnp.zeros((1, self.hidden_dim))
        
        for _ in range(self.num_layers):
            node_inputs = jnp.concatenate([V_pred, h], axis=-1)
            
            v_out, _ = SoftPhysicsGNNLayer(out_dim=self.hidden_dim)(node_inputs, senders, receivers, edge_features, edge_mask)
            h = nn.relu(v_out)
            
            dummy_graph_global = HyperHeteroMultiGraph(
                nodes={'bus': h},
                edges={('bus', 'line', 'bus'): EdgeIndices(senders=senders, receivers=receivers)},
                globals=global_state
            )
            super_node_layer = GlobalSuperNodeLayer(out_dim=self.hidden_dim)
            out_graph_global = super_node_layer(dummy_graph_global)
            
            h = out_graph_global.nodes['bus']
            global_state = out_graph_global.globals
            
            delta_V = nn.Dense(2)(h)
            V_pred = V_pred + delta_V
            
        return V_pred


class PowerFlowStrictSuperNodeGNN(nn.Module):
    hidden_dim: int = 32
    num_layers: int = 3
    
    @nn.compact
    def __call__(self, P_Q_inj, senders, receivers, edge_features, edge_mask=None):
        V_pred = jnp.zeros_like(P_Q_inj)
        V_pred = V_pred.at[:, 0].set(1.0)
        h = nn.Dense(self.hidden_dim)(P_Q_inj)
        
        global_state = jnp.zeros((1, self.hidden_dim))
        
        for _ in range(self.num_layers):
            node_inputs = jnp.concatenate([V_pred, h], axis=-1)
            
            v_out, _, _ = KVLOhmGNNLayer(out_dim=self.hidden_dim)(node_inputs, senders, receivers, edge_features)
            h = nn.relu(v_out)
            
            dummy_graph_global = HyperHeteroMultiGraph(
                nodes={'bus': h},
                edges={('bus', 'line', 'bus'): EdgeIndices(senders=senders, receivers=receivers)},
                globals=global_state
            )
            super_node_layer = GlobalSuperNodeLayer(out_dim=self.hidden_dim)
            out_graph_global = super_node_layer(dummy_graph_global)
            
            h = out_graph_global.nodes['bus']
            global_state = out_graph_global.globals
            
            delta_V = nn.Dense(2)(h)
            V_pred = V_pred + delta_V
            
        return V_pred


class PowerFlowUnconstrainedSuperNodeGNN(nn.Module):
    hidden_dim: int = 32
    num_layers: int = 3
    
    @nn.compact
    def __call__(self, P_Q_inj, senders, receivers, edge_features, edge_mask=None):
        V_pred = jnp.zeros_like(P_Q_inj)
        V_pred = V_pred.at[:, 0].set(1.0)
        h = nn.Dense(self.hidden_dim)(P_Q_inj)
        
        global_state = jnp.zeros((1, self.hidden_dim))
        
        for _ in range(self.num_layers):
            node_inputs = jnp.concatenate([V_pred, h], axis=-1)
            
            layer = UnconstrainedLayer(out_dim=self.hidden_dim)
            dummy_graph = HyperHeteroMultiGraph(
                nodes={'bus': node_inputs},
                edges={('bus', 'line', 'bus'): EdgeIndices(senders=senders, receivers=receivers)},
                edge_features={('bus', 'line', 'bus'): edge_features}
            )
            out_graph = layer(dummy_graph, edge_mask=edge_mask)
            h = nn.relu(out_graph.nodes['bus'])
            
            dummy_graph_global = HyperHeteroMultiGraph(
                nodes={'bus': h},
                edges={('bus', 'line', 'bus'): EdgeIndices(senders=senders, receivers=receivers)},
                globals=global_state
            )
            super_node_layer = GlobalSuperNodeLayer(out_dim=self.hidden_dim)
            out_graph_global = super_node_layer(dummy_graph_global)
            
            h = out_graph_global.nodes['bus']
            global_state = out_graph_global.globals
            
            delta_V = nn.Dense(2)(h)
            V_pred = V_pred + delta_V
            
        return V_pred
