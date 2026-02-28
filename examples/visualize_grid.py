import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse

def visualize_sample(sample, output_path="examples/power_grid_viz.png"):
    """
    Visualizes one power grid sample from the dataset.
    
    Sample dict format:
    - 'nodes': (N, 2) - [P_inj, Q_inj]
    - 'edges': {'senders': ..., 'receivers': ...}
    - 'edge_features': (E, 2) - [G, B]
    - 'labels': (N, 2) - [V_real, V_imag]
    """
    nodes = sample['nodes']
    edges = sample['edges']
    edge_features = sample['edge_features']
    labels = sample['labels']
    
    senders = edges['senders']
    receivers = edges['receivers']
    
    num_nodes = nodes.shape[0]
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes with features
    for i in range(num_nodes):
        # Calculate voltage magnitude from labels (V_real, V_imag)
        v_mag = np.sqrt(labels[i, 0]**2 + labels[i, 1]**2)
        p_inj = nodes[i, 0]
        q_inj = nodes[i, 1]
        
        G.add_node(i, p_inj=p_inj, q_inj=q_inj, v_mag=v_mag)
        
    # Add edges (only once for undirected visualization)
    for s, r, feat in zip(senders, receivers, edge_features):
        if s < r: 
            g, b = feat
            G.add_edge(s, r, g=g, b=b)
            
    # Layout selection based on size
    if num_nodes < 20:
        pos = nx.spring_layout(G, seed=42)
        node_size = 700
        with_labels = True
    else:
        pos = nx.kamada_kawai_layout(G)
        node_size = 50
        with_labels = False
    
    # Node colors based on Voltage Magnitude
    node_colors = [G.nodes[i]['v_mag'] for i in G.nodes]
    
    plt.figure(figsize=(12, 10))
    
    # Draw nodes
    nodes_plot = nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, 
                                       cmap=plt.cm.viridis, alpha=0.9)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.4, edge_color='gray')
    
    # Add labels if small enough
    if with_labels:
        labels_dict = {i: f"Bus {i}\nV:{G.nodes[i]['v_mag']:.3f}\nP:{G.nodes[i]['p_inj']:.2f}" for i in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels_dict, font_size=8, font_family="sans-serif")
    
    plt.title(f"Power Grid Visualization ({num_nodes} Nodes)")
    plt.colorbar(nodes_plot, label='Voltage Magnitude (p.u.)')
    plt.axis('off')
    
    # Add some text info
    plt.figtext(0.15, 0.05, f"Nodes: {num_nodes}, Edges: {G.number_of_edges()}", fontsize=10)
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize a power grid from the dataset.")
    parser.add_argument("--input", type=str, default="data/power_dataset.pkl", help="Path to the dataset pickle file.")
    parser.add_argument("--index", type=int, default=0, help="Index of the sample to visualize.")
    parser.add_argument("--output", type=str, default="examples/power_grid_viz.png", help="Path to save the visualization.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        sys.exit(1)
        
    with open(args.input, "rb") as f:
        dataset = pickle.load(f)
        
    print(f"Loaded dataset from {args.input} with {len(dataset)} samples.")
    
    if args.index >= len(dataset):
        print(f"Error: Index {args.index} out of range (dataset size {len(dataset)}).")
        sys.exit(1)
        
    # Visualize the selected sample
    sample = dataset[args.index]
    visualize_sample(sample, args.output)

if __name__ == "__main__":
    main()
