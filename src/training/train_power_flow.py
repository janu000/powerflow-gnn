import jax
import jax.numpy as jnp
import optax
import pickle
import flax.linen as nn
from flax.training import train_state
from typing import Any
import sys

from src.models.physics_layers import SoftPhysicsGNNLayer

class PowerFlowGNN(nn.Module):
    hidden_dim: int = 32
    num_layers: int = 3
    
    @nn.compact
    def __call__(self, P_Q_inj, senders, receivers, edge_features):
        # Start with flat voltage profile: V = 1.0 + j0.0
        V_pred = jnp.zeros_like(P_Q_inj)
        V_pred = V_pred.at[:, 0].set(1.0)
        
        # Embed node features (P, Q injections)
        h = nn.Dense(self.hidden_dim)(P_Q_inj)
        
        for _ in range(self.num_layers):
            # The physics layer needs V in the first 2 dimensions
            v_input = jnp.concatenate([V_pred, h], axis=-1)
            
            # Run soft physics layer
            v_out, _ = SoftPhysicsGNNLayer(out_dim=self.hidden_dim)(v_input, senders, receivers, edge_features)
            
            # Update node embeddings
            h = nn.relu(v_out)
            
            # Predict voltage residual (delta V)
            delta_V = nn.Dense(2)(h)
            V_pred = V_pred + delta_V
            
        return V_pred

def mse_loss(preds, targets):
    return jnp.mean((preds - targets) ** 2)

class TrainState(train_state.TrainState):
    pass

@jax.jit
def train_step(state: TrainState, batch: dict):
    nodes = batch['nodes']
    edges = batch['edges']
    edge_features = batch['edge_features']
    labels = batch['labels']
    senders, receivers = edges['senders'], edges['receivers']
    
    def loss_fn(params):
        preds = state.apply_fn(
            {'params': params}, 
            nodes, senders, receivers, edge_features
        )
        loss = mse_loss(preds, labels)
        return loss
        
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_step(state: TrainState, batch: dict):
    nodes = batch['nodes']
    edges = batch['edges']
    edge_features = batch['edge_features']
    labels = batch['labels']
    senders, receivers = edges['senders'], edges['receivers']
    
    preds = state.apply_fn(
        {'params': state.params}, 
        nodes, senders, receivers, edge_features
    )
    loss = mse_loss(preds, labels)
    return loss

def main():
    print("Loading dataset...")
    try:
        with open("data/power_dataset.pkl", "rb") as f:
            dataset = pickle.load(f)
    except FileNotFoundError:
        print("Dataset not found. Please run src/data/generate_power_dataset.py first.")
        sys.exit(1)
        
    # Split into train/val (80/20 split)
    train_data = dataset[:80]
    val_data = dataset[80:]
    
    # Initialize model
    rng = jax.random.PRNGKey(0)
    model = PowerFlowGNN(hidden_dim=32, num_layers=4)
    
    sample = train_data[0]
    variables = model.init(rng, sample['nodes'], sample['edges']['senders'], sample['edges']['receivers'], sample['edge_features'])
    
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optax.adam(1e-3)
    )
    
    epochs = 100
    print("Starting training (80/20 train/val split)...")
    
    for epoch in range(epochs):
        # Training
        epoch_train_loss = 0.0
        for batch in train_data:
            state, loss = train_step(state, batch)
            epoch_train_loss += loss
            
        # Validation
        epoch_val_loss = 0.0
        for batch in val_data:
            val_loss = eval_step(state, batch)
            epoch_val_loss += val_loss
            
        if (epoch + 1) % 10 == 0:
            avg_train_loss = epoch_train_loss / len(train_data)
            avg_val_loss = epoch_val_loss / len(val_data)
            print(f"Epoch {epoch + 1:3d}/{epochs} - Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

if __name__ == "__main__":
    main()
