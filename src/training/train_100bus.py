import jax
import optax
from flax.training import train_state
import sys
import time
from tabulate import tabulate
from tqdm import tqdm

from src.data.dataloader import GenericDataLoader
from src.models import (
    PowerFlowSoftGNN,
    PowerFlowStrictGNN,
    PowerFlowUnconstrainedGNN
)
from src.training.metrics import mse_loss, calculate_metrics

class TrainState(train_state.TrainState):
    pass

def train_model(model_class, splits, epochs=200, name="Model"):
    rng = jax.random.PRNGKey(42)
    model = model_class()
    
    train_data = splits['train']
    val_interp = splits['test_interpolation']
    val_extrap = splits['test_extrapolation']
    
    sample = train_data[0]
    variables = model.init(rng, sample['nodes'], sample['edges']['senders'], sample['edges']['receivers'], sample['edge_features'], sample['edge_mask'])
    
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
                batch['nodes'], batch['edges']['senders'], batch['edges']['receivers'], batch['edge_features'], batch['edge_mask']
            )
            return mse_loss(preds, batch['labels'], batch['node_mask'])
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    @jax.jit
    def eval_step(state, batch):
        preds = state.apply_fn(
            {'params': state.params}, 
            batch['nodes'], batch['edges']['senders'], batch['edges']['receivers'], batch['edge_features'], batch['edge_mask']
        )
        return calculate_metrics(preds, batch)

    print(f"\nTraining {name}...")
    start_time = time.time()
    
    history = {k: [] for k in ['mse', 'vol_acc', 'p_mismatch', 'q_mismatch', 'feasible']}
    
    for epoch in tqdm(range(epochs), desc=f"{name} Epochs"):
        for batch in train_data:
            state, loss = train_step(state, batch)
            
        # Track convergence on the interpolation validation set
        agg_metrics = {k: 0.0 for k in history.keys()}
        for batch in val_interp:
            m = eval_step(state, batch)
            for k in agg_metrics:
                agg_metrics[k] += m[k]
        
        n = len(val_interp)
        for k in agg_metrics:
            history[k].append(float(agg_metrics[k] / n))
            
    print(f"{name} finished training in {time.time() - start_time:.2f}s")
    
    # Evaluate
    def evaluate_split(split_data, desc):
        agg_metrics = {'mse': 0.0, 'vol_acc': 0.0, 'p_mismatch': 0.0, 'q_mismatch': 0.0, 'feasible': 0.0}
        for batch in tqdm(split_data, desc=desc, leave=False):
            m = eval_step(state, batch)
            for k in agg_metrics:
                agg_metrics[k] += m[k]
        
        n = len(split_data)
        return {k: float(v / n) for k, v in agg_metrics.items()}
        
    res_interp = evaluate_split(val_interp, desc="Eval Interp")
    res_extrap = evaluate_split(val_extrap, desc="Eval Extrap")
    
    return {
        'model': name,
        'interp': res_interp,
        'extrap': res_extrap,
        'history': history
    }

def main():
    print("Loading 100-bus dataset...")
    try:
        loader = GenericDataLoader("data/power_dataset_100bus.pkl")
    except FileNotFoundError:
        print("Dataset not found. Run the generation script first.")
        sys.exit(1)
        
    splits = loader.get_bus_size_splits()
    print(f"Data Splits: {len(splits['train'])} Train | {len(splits['test_interpolation'])} Val (Interp) | {len(splits['test_extrapolation'])} Val (Extrap)")
    
    results = []
    results.append(train_model(PowerFlowSoftGNN, splits, epochs=600, name="Soft Physics GNN"))
    results.append(train_model(PowerFlowStrictGNN, splits, epochs=600, name="Strict Physics GNN"))
    results.append(train_model(PowerFlowUnconstrainedGNN, splits, epochs=600, name="Unconstrained GNN"))
    
    # Format and save results
    md_content = "# Training Results on 100-Bus Systems\n\n"
    
    for split_type, split_name in [('interp', 'Interpolation (93-107 buses)'), ('extrap', 'Extrapolation (90-92 & 108-110 buses)')]:
        md_content += f"### {split_name}\n\n"
        
        table = []
        headers = ["Model", "MSE Loss", "Vol Accuracy", "P Mismatch", "Q Mismatch", "Feasibility Rate"]
        
        for r in results:
            m = r[split_type]
            table.append([
                r['model'], 
                f"{m['mse']:.6f}", 
                f"{m['vol_acc']:.6f}", 
                f"{m['p_mismatch']:.6f}", 
                f"{m['q_mismatch']:.6f}", 
                f"{m['feasible']*100:.1f}%"
            ])
            
        md_table = tabulate(table, headers, tablefmt="github")
        md_content += md_table + "\n\n"
        print(f"\n{split_name}")
        print(md_table)
        
    with open("training_results.md", "w") as f:
        f.write(md_content)
        
    print("\nResults saved to training_results.md")

    # Plotting Training Convergence
    print("\nGenerating convergence plots...")
    import matplotlib.pyplot as plt
    metrics = ['mse', 'vol_acc', 'p_mismatch', 'q_mismatch', 'feasible']
    metric_names = ['MSE Loss', 'Voltage Accuracy', 'P Mismatch', 'Q Mismatch', 'Feasibility Rate']
    
    for metric, m_name in zip(metrics, metric_names):
        plt.figure(figsize=(10, 6))
        for r in results:
            plt.plot(r['history'][metric], label=r['model'])
        plt.title(f'Validation Convergence: {m_name}')
        plt.xlabel('Epoch')
        plt.ylabel(m_name)
        if metric in ['mse', 'vol_acc', 'p_mismatch', 'q_mismatch']:
            plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"plot_{metric}_convergence.png")
        plt.close()
        
    print("Plots saved as plot_*.png")

if __name__ == "__main__":
    main()
