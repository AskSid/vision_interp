import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb
import itertools
import csv
from src.vision_interp.SAE import VanillaSAE, BatchTopKSAE, TopKSAE, DeadLatentTracker
from src.utils import set_seed, load_saved_vision_model_activations

def train_SAE(model, activation_dataloader, config, device):
    """Train a single SAE model"""
    if config['wandb']:
        # Use wandb_project from config if specified, otherwise fall back to default naming
        if 'wandb_project' in config:
            project_name = config['wandb_project']
        else:
            project_name = f"{config['dataset']}-{config['model_name']}-{config['experiment_name']}-SAE"
        
        wandb.init(project=project_name, 
                  config=config,
                  name=config['sae_name'],
                  reinit=True)
    
    optimizer = optim.Adam(model.parameters(), lr=float(config['lr']), weight_decay=float(config['weight_decay']))
    best_loss = float('inf')
    
    for epoch in tqdm(range(int(config['epochs'])), desc="Training epochs"):
        dead_latent_tracker = DeadLatentTracker(model.config['activation_size'] * model.config['expansion_factor'], device)
        epoch_losses = []
        
        for activations_batch in activation_dataloader:
            x = activations_batch[0].to(device)
            x_hat, hidden = model(x)
            dead_latent_tracker.update(hidden)
            loss, loss_dict = model.compute_loss(x, x_hat, hidden)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss_dict)

        avg_loss = sum(l['total_loss'] for l in epoch_losses) / len(epoch_losses)
        avg_recon = sum(l['l2'] for l in epoch_losses) / len(epoch_losses)
        avg_l1 = sum(l['l1'] for l in epoch_losses) / len(epoch_losses)
        avg_l0_norm = sum(l['l0'] for l in epoch_losses) / len(epoch_losses)
        avg_r2 = sum(l['r2'] for l in epoch_losses) / len(epoch_losses)
        fraction_alive = dead_latent_tracker.get_fraction_alive()
        
        if config['wandb']:
            wandb.log({
                "epoch": epoch,
                "total_loss": avg_loss,
                "l2": avg_recon,
                "l1": avg_l1,
                "l0": avg_l0_norm,
                "r2": avg_r2,
                "fraction_alive": fraction_alive
            })
        
        print(f'Epoch {epoch+1}/{config["epochs"]}:', flush=True)
        print(f'  Total Loss: {avg_loss:.6f}, Reconstruction Loss: {avg_recon:.6f}, L1 Loss: {avg_l1:.6f}, L0 Norm: {avg_l0_norm}, R2: {avg_r2:.3f}, Fraction Alive: {fraction_alive:.3f}', flush=True)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_stats = {
                "loss": avg_loss,
                "l2": avg_recon,
                "l1": avg_l1,
                "l0": avg_l0_norm,
                "r2": avg_r2,
                "fraction_alive": fraction_alive
            }
            
            # Save model and config
            os.makedirs(config['sae_dir'], exist_ok=True)
            torch.save(model.state_dict(), config['sae_weights_path'])
            print(f"Saved SAE to {config['sae_weights_path']}", flush=True)
            
            # Save best stats to config
            config['best_stats'] = best_stats
            with open(config['sae_config_path'], 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
    
    if config['wandb']:
        wandb.finish()
    return best_stats

def run_single_training(config, device):
    """Run a single training run with the given config"""
    print("Training configuration:")
    for key, value in config.items():
        if key in ['sae_type', 'expansion_factor', 'l1_coeff', 'topk', 'batch_size', 'lr', 'weight_decay', 'epochs']:
            print(f"  {key}: {value}")
    print()
    
    activations, _, _ = load_saved_vision_model_activations(config['data_dir'], config['n_samples_per_image'])
    activation_dataset = TensorDataset(activations)
    activation_dataloader = DataLoader(activation_dataset, batch_size=int(config['batch_size']), shuffle=True)
    print(f"\nLoaded {activations.shape[0]} activation vectors of size {activations.shape[1]}.\n")
    
    sae_model_config = {
        'activation_size': activations.shape[1],
        'expansion_factor': config['expansion_factor'],
        'l1_coeff': config['l1_coeff'],
        'topk': config['topk'],
        'training': True
    }
    
    sae_models = {
        "vanilla": VanillaSAE,
        "batch_topk": BatchTopKSAE,
        "topk": TopKSAE
    }
    sae_model = sae_models[config['sae_type']](config=sae_model_config).to(device)
    print(f"Training {config['sae_type']} SAE on {activations.shape[1]}-dimensional activations.")
    print(f"SAE will have {activations.shape[1] * config['expansion_factor']} latents.\n")
    
    # Save the config before training
    os.makedirs(config['sae_dir'], exist_ok=True)
    with open(config['sae_config_path'], 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return train_SAE(sae_model, activation_dataloader, config, device)

def setup_paths(config, is_sweep=False):
    """Set up directory and file paths for the SAE"""
    base_models_dir = config.get('models_dir', "src/vision_interp/models")
    
    # Generate SAE name if not provided
    if not config.get('sae_name'):
        config['sae_name'] = generate_sae_name(config)
    
    # Set up directory structure
    if is_sweep or is_sweep_config(config):
        # For sweeps: models/sweep_dir/sae_dir/
        config['sae_dir'] = os.path.join(
            base_models_dir, 
            config['experiment_name'], 
            config['sae_name']
        )
    else:
        # For single runs: models/sae_dir/
        config['sae_dir'] = os.path.join(
            base_models_dir, 
            config['sae_name']
        )
    
    # Set up file paths within the SAE directory
    config['sae_weights_path'] = os.path.join(config['sae_dir'], 'sae.pth')
    config['sae_config_path'] = os.path.join(config['sae_dir'], 'config.yaml')
    config['activations_path'] = os.path.join(config['sae_dir'], 'sae_activations.pt')
    config['activating_examples_dir'] = os.path.join(config['sae_dir'], 'activating_examples')

def generate_sae_name(config):
    """Generate a descriptive SAE name based on config parameters"""
    parts = [
        config['sae_type'],
        f"exp{config['expansion_factor']}",
        f"l1{config['l1_coeff']}",
    ]
    
    if config['sae_type'] == 'topk':
        parts.append(f"topk{config['topk']}")
    
    parts.extend([
        f"lr{config['lr']}",
        f"ep{config['epochs']}",
        config['dataset'],
        config['model_name'],
        config['layer_name'].replace('.', '')
    ])
    
    return "_".join(parts)

def is_sweep_config(config):
    """Check if this config contains sweep parameters (lists)"""
    for value in config.values():
        if isinstance(value, list):
            return True
    return False

def get_sweep_combinations(config):
    """Generate all combinations for a sweep config"""
    if not is_sweep_config(config):
        return [config]
    
    # Find sweep parameters
    sweep_params = {}
    for key, value in config.items():
        if isinstance(value, list):
            sweep_params[key] = value
    
    # Generate all combinations
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    combinations = list(itertools.product(*param_values))
    
    # Create config for each combination
    configs = []
    for combination in combinations:
        combo_config = config.copy()
        sae_name_parts = []
        
        for param_name, param_value in zip(param_names, combination):
            combo_config[param_name] = param_value
            sae_name_parts.append(f"{param_name}-{param_value}")
        
        # Generate unique SAE name for this combination
        combo_config['sae_name'] = f"{config['sae_name']}_{'_'.join(sae_name_parts)}"
        
        # Set up paths for this combination
        setup_paths(combo_config, is_sweep=True)
        configs.append(combo_config)
    
    return configs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="path to the unified config file")
    args = ap.parse_args()
    
    # Load the config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}.")

    set_seed(config['seed'])
    
    if is_sweep_config(config):
        print("Sweep parameters detected:")
        sweep_params = {}
        for key, value in config.items():
            if isinstance(value, list):
                sweep_params[key] = value
                print(f"  {key}: {value}")
        print()
        
        # Get all sweep combinations
        sweep_configs = get_sweep_combinations(config)
        print(f"Total combinations: {len(sweep_configs)}")
        print()
        
        # Create sweep directory under models and CSV file
        sweep_dir = os.path.join(config.get('models_dir', "src/vision_interp/models"), config['experiment_name'])
        os.makedirs(sweep_dir, exist_ok=True)
        
        # Define all possible columns (parameters + stats)
        all_param_names = ['sae_type', 'expansion_factor', 'l1_coeff', 'topk', 'batch_size', 'lr', 'weight_decay', 'epochs', 'n_samples_per_image']
        stat_names = ['loss', 'l2', 'l1', 'l0', 'r2', 'fraction_alive']
        fieldnames = all_param_names + stat_names
        
        # Create CSV file with headers in sweep_dir
        csv_path = os.path.join(sweep_dir, f"{config['experiment_name']}.csv")
        with open(csv_path, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        
        # Run each combination
        for i, combo_config in enumerate(sweep_configs):
            print(f"Running combination {i+1}/{len(sweep_configs)}:")
            for param_name, param_value in combo_config.items():
                if param_name in sweep_params:
                    print(f"  {param_name}: {param_value}")
            
            # Run training
            best_stats = run_single_training(combo_config, device)
            print(f"Combination {i+1} finished training.")
            print()
            
            # Save results to CSV
            with open(csv_path, mode='a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                row = {}
                for param in all_param_names:
                    row[param] = combo_config.get(param, None)
                for stat in stat_names:
                    row[stat] = best_stats.get(stat, None)
                writer.writerow(row)
        
        print(f"Sweep completed! Results saved in {csv_path}")
        
    else:
        # Single training run
        print("Single training run configuration")
        setup_paths(config)
        run_single_training(config, device)

if __name__ == '__main__':
    main()
