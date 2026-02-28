import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def plot_distribution(dataset_path="data/power_dataset_100bus.pkl", output_path="examples/bus_distribution.png"):
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        sys.exit(1)
        
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
        
    print(f"Loaded dataset from {dataset_path} with {len(dataset)} samples.")
    
    # Extract number of buses from each sample
    # The 'num_nodes' key was added during generation
    num_buses = [sample['num_nodes'] for sample in dataset]
    
    plt.figure(figsize=(10, 6))
    
    # Create histogram with bins for each possible integer in the range 90-110
    bins = np.arange(min(num_buses) - 0.5, max(num_buses) + 1.5, 1)
    
    n, bins, patches = plt.hist(num_buses, bins=bins, rwidth=0.8, color='skyblue', edgecolor='black', alpha=0.7)
    
    plt.xticks(np.arange(min(num_buses), max(num_buses) + 1, 1))
    plt.xlabel('Number of Buses')
    plt.ylabel('Frequency')
    plt.title('Distribution of Grid Sizes in 100-Bus Dataset')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add counts on top of bars
    for i in range(len(n)):
        if n[i] > 0:
            plt.text(bins[i] + 0.5, n[i] + 0.2, int(n[i]), ha='center', va='bottom', fontsize=9)
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Distribution plot saved to {output_path}")

if __name__ == "__main__":
    plot_distribution()
