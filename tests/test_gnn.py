import jax
import jax.numpy as jnp
import chex
from src.models.gnn import HyperHeteroGNN
from src.data.dataset import generate_dummy_graph

def test_gnn_forward():
    key = jax.random.PRNGKey(0)
    model = HyperHeteroGNN(
        hidden_dims=[16, 16],
        out_dim=2,
        num_layers=2
    )
    
    graph = generate_dummy_graph(key, num_nodes_per_type=5)
    variables = model.init(key, graph)
    
    logits = model.apply(variables, graph)
    
    # Check output shape (graph classification assumption)
    chex.assert_shape(logits, (2,))
