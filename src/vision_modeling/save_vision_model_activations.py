import os
import argparse
import yaml
from typing import Dict, List
import torch
from torch import nn
from einops import rearrange
from src.activation_saver import get_model_activations
from src.utils import load_model, load_vision_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="path to the config file")
    args = ap.parse_args()
    
    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}.")

    train_loader, _, num_classes, _ = load_vision_dataset(config['dataset'], config['batch_size'], config['num_workers'], shuffle_train=False)
    config['num_classes'] = num_classes
    model = load_model(model_config=config, device=device)

    for layer_name in config['layer_names']:
        print(f"\nProcessing layer: {layer_name}")
        
        os.makedirs(f"{config['save_dir']}/{config['dataset']}_{config['model_name']}/{layer_name.replace('.', '')}", exist_ok=True)
        for i, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(device)
            dataset_indices = torch.arange(i * config['batch_size'], (i * config['batch_size']) + inputs.shape[0])
            
            activations = get_model_activations(model, inputs, [layer_name])
            activation_tensor = activations[layer_name]  # (B, C, H, W)
            b, c, h, w = activation_tensor.shape
            
            activation_flat = rearrange(activation_tensor, 'b c h w -> (b h w) c')
            
            # Create position indices for all spatial locations
            h_coords, w_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            h_coords = h_coords.flatten().repeat(b)  # [b * h * w]
            w_coords = w_coords.flatten().repeat(b)  # [b * h * w]
            position_indices = torch.stack([h_coords, w_coords], dim=1)  # [b * h * w, 2]
            
            # Repeat dataset indices for each spatial location
            dataset_indices_flat = dataset_indices.repeat_interleave(h * w)
            
            save_path = f"{config['save_dir']}/{config['dataset']}_{config['model_name']}/{layer_name.replace('.', '')}/batch_{i}.pt"
            torch.save({
                'activations': activation_flat,
                'dataset_indices': dataset_indices_flat,
                'image_indices': position_indices
            }, save_path)
            
            if i == 0:
                print(f"  {layer_name}: {activations[layer_name].shape}")
            
            del activations, inputs, dataset_indices
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1} batches for {layer_name}...")
        
        print(f"Completed layer: {layer_name}, saved to {save_path}")
    
    print(f"\nSaved activations for all {len(config['layer_names'])} layers.")

if __name__ == '__main__':
    main()