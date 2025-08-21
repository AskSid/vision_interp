import os
import sys
import argparse
import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.vision_interp.SAE import VanillaSAE, BatchTopKSAE, TopKSAE
from src.utils import set_seed, load_saved_vision_model_activations


def get_data_and_sae(config, device):
    """Load activations and SAE model using config"""
    activations, dataset_indices, image_indices = load_saved_vision_model_activations(
        config['data_dir'], 
        config['n_samples_per_image']
    )
    activation_dataset = TensorDataset(activations)
    activation_dataloader = DataLoader(
        activation_dataset, 
        batch_size=int(config['activation_batch_size']), 
        shuffle=False
    )
    print(f"Loaded {activations.shape[0]} model activations pre-SAE.")
    
    sae_model_config = {
        'activation_size': activations.shape[1],
        'expansion_factor': config['expansion_factor'],
        'l1_coeff': config['l1_coeff'],
        'topk': config['topk'],
        'training': False
    }
    
    sae_models = {
        "vanilla": VanillaSAE,
        "batch_topk": BatchTopKSAE,
        "topk": TopKSAE
    }
    model = sae_models[config['sae_type']](config=sae_model_config)
    model.load_state_dict(torch.load(config['sae_weights_path'], map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded SAE model from {config['sae_weights_path']}")

    return model, activation_dataloader, dataset_indices, image_indices



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-dir", required=True, help="Path to SAE directory containing config.yaml and sae.pth")
    args = ap.parse_args()
    
    # Load config from SAE directory
    config_path = os.path.join(args.sae_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from SAE directory: {args.sae_dir}")
    
    set_seed(config.get('seed', 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}.")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(config['activations_path']), exist_ok=True)
    
    model, activation_dataloader, dataset_indices, image_indices = get_data_and_sae(config, device)

    all_activations = []
    for batch_idx, inputs in enumerate(activation_dataloader):
        x = inputs[0].to(device)
        with torch.no_grad():
            activations = model.encode(x)

        all_activations.append(activations.cpu())
        if batch_idx == 0:
            print(f"Processed batch {batch_idx + 1} with input shape {x.shape} and output shape {activations.shape}.")
    
    all_activations = torch.cat(all_activations, dim=0)
    activations_path = Path(config['activations_path'])
    # Save activations
    torch.save({
        'activations': all_activations,
        'dataset_indices': dataset_indices,
        'image_indices': image_indices
    }, activations_path)
    
    print(f"\nSaved complete mapping to {activations_path}. {all_activations.shape[0]} activations saved.")
    


if __name__ == "__main__":
    main()
