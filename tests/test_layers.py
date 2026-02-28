import jax
import jax.numpy as jnp
import chex
from src.data.graph import HyperHeteroMultiGraph, EdgeIndices
from src.models.layers import HeteroMessagePassing

def test_hetero_message_passing_forward():
    key = jax.random.PRNGKey(0)
    
    nodes = {
        'a': jnp.ones((2, 8)),
        'b': jnp.ones((3, 8))
    }
    edges = {
        ('a', 'to', 'b'): EdgeIndices(
            senders=jnp.array([0, 1]),
            receivers=jnp.array([0, 1])
        )
    }
    n_node = {'a': jnp.array([2]), 'b': jnp.array([3])}
    n_edge = {('a', 'to', 'b'): jnp.array([2])}
    
    graph = HyperHeteroMultiGraph(
        nodes=nodes,
        edges=edges,
        n_node=n_node,
        n_edge=n_edge
    )
    
    layer = HeteroMessagePassing(out_dim=16)
    variables = layer.init(key, graph)
    
    out_graph = layer.apply(variables, graph)
    
    # Check that node features have been updated to the correct dimension
    for ntype in out_graph.nodes:
        chex.assert_shape(out_graph.nodes[ntype], (graph.nodes[ntype].shape[0], 16))
