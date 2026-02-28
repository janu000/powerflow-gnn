import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from src.models.gnn import HyperHeteroGNN
from src.training.metrics import cross_entropy_loss, accuracy
from typing import Tuple, Any

class TrainState(train_state.TrainState):
    """Custom train state if needed for batch stats."""
    pass

def create_train_state(rng, model, sample_graph, learning_rate):
    """Creates initial `TrainState`."""
    params = model.init(rng, sample_graph)['params']
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state: TrainState, batch: Tuple[Any, jnp.ndarray]) -> Tuple[TrainState, float, float]:
    """Trains for a single step."""
    graph, label = batch
    
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, graph)
        loss = cross_entropy_loss(logits, label)
        return loss, logits
        
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    
    state = state.apply_gradients(grads=grads)
    acc = accuracy(logits, label)
    
    return state, loss, acc

@jax.jit
def eval_step(state: TrainState, batch: Tuple[Any, jnp.ndarray]) -> Tuple[float, float]:
    """Evaluates for a single step."""
    graph, label = batch
    logits = state.apply_fn({'params': state.params}, graph)
    loss = cross_entropy_loss(logits, label)
    acc = accuracy(logits, label)
    return loss, acc
