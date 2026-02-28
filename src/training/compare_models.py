import jax
import jax.numpy as jnp
import optax
import pickle
import flax.linen as nn
from flax.training import train_state
from typing import Any
import sys
import time

from src.models.physics_layers import SoftPhysicsGNNLayer, KVLOhmGNNLayer
from src.models.layers import BaseHeteroMessagePassing

# --- 1. Soft Physics GNN ---
class PowerFlowSoftGNN(nn.Module):
    hidden_dim: int = 32
    num_layers: int = 3
    
    @nn.compact
    def __call__(self, P_Q_inj, senders, receivers, edge_features):
        V_pred = jnp.zeros_like(P_Q_inj)
        V_pred = V_pred.at[:, 0].set(1.0)
        h = nn.Dense(self.hidden_dim)(P_Q_inj)
        
        for _ in range(self.num_layers):
            v_input = jnp.concatenate([V_pred, h], axis=-1)
            v_out, _ = SoftPhysicsGNNLayer(out_dim=self.hidden_dim)(v_input, senders, receivers, edge_features)
            h = nn.relu(v_out)
            delta_V = nn.Dense(2)(h)
            V_pred = V_pred + delta_V
            
        return V_pred

# --- 2. Strict Physics GNN (KVL + Ohm) ---
class PowerFlowStrictGNN(nn.Module):
    hidden_dim: int = 32
    num_layers: int = 3
    
    @nn.compact
    def __call__(self, P_Q_inj, senders, receivers, edge_features):
        V_pred = jnp.zeros_like(P_Q_inj)
        V_pred = V_pred.at[:, 0].set(1.0)
        h = nn.Dense(self.hidden_dim)(P_Q_inj)
        
        for _ in range(self.num_layers):
            # Strict layer needs the exact voltage prediction in the first 2 dims
            v_input = jnp.concatenate([V_pred, h], axis=-1)
            
            # The strict layer calculates V_edge and I_edge purely via math
            # and only uses the NN to update the node states.
            v_out, _, _ = KVLOhmGNNLayer(out_dim=self.hidden_dim)(v_input, senders, receivers, edge_features)
            
            h = nn.relu(v_out)
            delta_V = nn.Dense(2)(h)
            V_pred = V_pred + delta_V
            
        return V_pred

# --- 3. Unconstrained Baseline GNN ---
class UnconstrainedLayer(BaseHeteroMessagePassing):
    @nn.compact
    def message(self, edge_type, src_feats, dst_feats, edge_feats=None):
        msg_inputs = [src_feats]
        if edge_feats is not None:
            msg_inputs.append(edge_feats)
        return nn.Dense(self.out_dim)(jnp.concatenate(msg_inputs, axis=-1))

class PowerFlowUnconstrainedGNN(nn.Module):
    hidden_dim: int = 32
    num_layers: int = 3
    
    @nn.compact
    def __call__(self, P_Q_inj, senders, receivers, edge_features):
        V_pred = jnp.zeros_like(P_Q_inj)
        V_pred = V_pred.at[:, 0].set(1.0)
        h = nn.Dense(self.hidden_dim)(P_Q_inj)
        
        for _ in range(self.num_layers):
            # Same iterative structure, but purely neural
            node_inputs = jnp.concatenate([V_pred, h], axis=-1)
            
            # Use the unconstrained message passing
            layer = UnconstrainedLayer(out_dim=self.hidden_dim)
            
            # Create a dummy edges dict to fit the base class API
            from src.data.graph import HyperHeteroMultiGraph, EdgeIndices
            dummy_graph = HyperHeteroMultiGraph(
                nodes={'bus': node_inputs},
                edges={('bus', 'line', 'bus'): EdgeIndices(senders=senders, receivers=receivers)},
                edge_features={('bus', 'line', 'bus'): edge_features}
            )
            
            out_graph = layer(dummy_graph)
            h = out_graph.nodes['bus']
            
            delta_V = nn.Dense(2)(h)
            V_pred = V_pred + delta_V
            
        return V_pred

def mse_loss(preds, targets):
    return jnp.mean((preds - targets) ** 2)

class TrainState(train_state.TrainState):
    pass

def train_model(model_class, train_data, val_data, epochs=100, name="Model"):
    rng = jax.random.PRNGKey(0)
    model = model_class()
    
    sample = train_data[0]
    variables = model.init(rng, sample['nodes'], sample['edges']['senders'], sample['edges']['receivers'], sample['edge_features'])
    
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optax.adam(1e-3)
    )
    
    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            preds = state.apply_fn(
                {'params': params}, 
                batch['nodes'], batch['edges']['senders'], batch['edges']['receivers'], batch['edge_features']
            )
            return mse_loss(preds, batch['labels'])
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    @jax.jit
    def eval_step(state, batch):
        preds = state.apply_fn(
            {'params': state.params}, 
            batch['nodes'], batch['edges']['senders'], batch['edges']['receivers'], batch['edge_features']
        )
        return mse_loss(preds, batch['labels'])

    print(f"\nTraining {name}...")
    start_time = time.time()
    
    for epoch in range(epochs):
        train_loss = 0.0
        for batch in train_data:
            state, loss = train_step(state, batch)
            train_loss += loss
            
        val_loss = 0.0
        for batch in val_data:
            val_loss += eval_step(state, batch)
            
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1:3d} | Train Loss: {train_loss / len(train_data):.6f} | Val Loss: {val_loss / len(val_data):.6f}")
            
    print(f"{name} finished in {time.time() - start_time:.2f}s")
    return state, val_loss / len(val_data)

def main():
    try:
        with open("data/power_dataset.pkl", "rb") as f:
            dataset = pickle.load(f)
    except FileNotFoundError:
        print("Dataset not found.")
        sys.exit(1)
        
    train_data = dataset[:80]
    val_data = dataset[80:]
    
    print(f"Dataset: {len(train_data)} train, {len(val_data)} val")
    
    # Train all 3 models
    _, val_loss_soft = train_model(PowerFlowSoftGNN, train_data, val_data, epochs=150, name="SoftPhysicsGNN")
    _, val_loss_strict = train_model(PowerFlowStrictGNN, train_data, val_data, epochs=150, name="StrictPhysicsGNN")
    _, val_loss_unconstrained = train_model(PowerFlowUnconstrainedGNN, train_data, val_data, epochs=150, name="UnconstrainedGNN")
    
    print("\n--- FINAL VALIDATION COMPARISON ---")
    print(f"Strict Physics GNN Loss : {val_loss_strict:.6f}")
    print(f"Soft Physics GNN Loss   : {val_loss_soft:.6f}")
    print(f"Unconstrained GNN Loss  : {val_loss_unconstrained:.6f}")

if __name__ == "__main__":
    main()
