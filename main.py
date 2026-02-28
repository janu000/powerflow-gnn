import jax
import argparse
import sys
from tqdm import tqdm
from src.utils.config import load_config
from src.data.dataset import DummyDataset, generate_dummy_graph
from src.models.gnn import HyperHeteroGNN
from src.training.trainer import create_train_state, train_step

def main(config_path: str):
    config = load_config(config_path)
    
    print("Initializing JAX environment...")
    print(f"Devices available: {jax.devices()}")
    
    # Initialize Dataset
    dataset = DummyDataset(size=config['data']['num_samples'])
    
    # Dummy sample for initialization
    init_rng = jax.random.PRNGKey(0)
    sample_graph = generate_dummy_graph(init_rng)
    
    # Initialize Model
    model = HyperHeteroGNN(
        hidden_dims=config['model']['hidden_dims'],
        out_dim=config['model']['out_dim'],
        num_layers=config['model']['num_layers']
    )
    
    # Initialize Training State
    state = create_train_state(
        init_rng, 
        model, 
        sample_graph, 
        config['training']['learning_rate']
    )
    
    print("Starting Training Loop...")
    epochs = config['training']['epochs']
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        steps = 0
        
        # Simulating batches
        for graph, label in tqdm(dataset, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = (graph, label)
            state, loss, acc = train_step(state, batch)
            
            epoch_loss += loss
            epoch_acc += acc
            steps += 1
            
        print(f"Epoch {epoch+1} | Loss: {epoch_loss/steps:.4f} | Acc: {epoch_acc/steps:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Hyper Heterogeneous Multi-Graph GNN")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to the config file")
    args = parser.parse_args()
    
    main(args.config)
