import jax
import optax
from flax.training import train_state
from typing import Dict
from tqdm import tqdm

from src.data.dataloader import GenericDataLoader
from src.models import (
    PowerFlowSoftSuperNodeGNN,
    PowerFlowStrictSuperNodeGNN,
    PowerFlowUnconstrainedSuperNodeGNN
)
from src.training.metrics import mse_loss, calculate_metrics

class TrainState(train_state.TrainState):
    pass

def train_model(model_class, splits, epochs=600, name="Model"):
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
    
    for epoch in tqdm(range(epochs), desc=f"{name} Epochs"):
        for batch in train_data:
            state, loss = train_step(state, batch)
            
    def evaluate_split(split_data):
        agg_metrics = {'mse': 0.0, 'vol_acc': 0.0, 'p_mismatch': 0.0, 'q_mismatch': 0.0, 'feasible': 0.0}
        for batch in split_data:
            m = eval_step(state, batch)
            for k in agg_metrics:
                agg_metrics[k] += m[k]
        
        n = len(split_data)
        return {k: float(v / n) for k, v in agg_metrics.items()}
        
    res_interp = evaluate_split(val_interp)
    res_extrap = evaluate_split(val_extrap)
    
    return {'model': name, 'interp': res_interp, 'extrap': res_extrap}

def main():
    loader = GenericDataLoader("data/power_dataset_100bus.pkl")
    splits = loader.get_bus_size_splits()
    
    results = []
    results.append(train_model(PowerFlowSoftSuperNodeGNN, splits, epochs=600, name="Soft Physics Global GNN"))
    results.append(train_model(PowerFlowStrictSuperNodeGNN, splits, epochs=600, name="Strict Physics Global GNN"))
    results.append(train_model(PowerFlowUnconstrainedSuperNodeGNN, splits, epochs=600, name="Unconstrained Global GNN"))
    
    # Append to markdown file
    with open("training_results.md", "r") as f:
        content = f.read()
        
    parts = content.split("### Extrapolation (90-92 & 108-110 buses)")
    interp_section = parts[0].strip()
    extrap_section = "### Extrapolation (90-92 & 108-110 buses)\n\n" + parts[1].strip()
    
    # Remove trailing newlines from the table block to append nicely
    interp_section = interp_section.rstrip()
    
    new_interp_rows = ""
    new_extrap_rows = ""
    
    for r in results:
        m_i = r['interp']
        m_e = r['extrap']
        new_interp_rows += f"\n| {r['model']:<25} | {m_i['mse']:.6f} | {m_i['vol_acc']:.6f} | {m_i['p_mismatch']:.6f} | {m_i['q_mismatch']:.6f} | {m_i['feasible']*100:.1f}% |"
        new_extrap_rows += f"\n| {r['model']:<25} | {m_e['mse']:.6f} | {m_e['vol_acc']:.6f} | {m_e['p_mismatch']:.6f} | {m_e['q_mismatch']:.6f} | {m_e['feasible']*100:.1f}% |"

    new_content = interp_section + new_interp_rows + "\n\n" + extrap_section + new_extrap_rows + "\n"
    
    with open("training_results.md", "w") as f:
        f.write(new_content)
    
    print("\nResults appended to training_results.md")

if __name__ == "__main__":
    main()
