import jax
import jax.numpy as jnp
from src.data.graph import HyperHeteroMultiGraph, EdgeIndices

def test_graph_initialization():
    nodes = {
        'user': jnp.zeros((5, 16)),
        'item': jnp.zeros((10, 32))
    }
    edges = {
        ('user', 'buys', 'item'): EdgeIndices(
            senders=jnp.array([0, 1, 2]),
            receivers=jnp.array([5, 6, 7])
        )
    }
    n_node = {'user': jnp.array([5]), 'item': jnp.array([10])}
    n_edge = {('user', 'buys', 'item'): jnp.array([3])}
    
    graph = HyperHeteroMultiGraph(
        nodes=nodes,
        edges=edges,
        n_node=n_node,
        n_edge=n_edge
    )
    
    assert graph.nodes['user'].shape == (5, 16)
    assert graph.edges[('user', 'buys', 'item')].senders.shape == (3,)
