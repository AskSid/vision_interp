import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torchvision import datasets, transforms
from tqdm import tqdm
from collections import defaultdict
import argparse
import yaml
from einops import rearrange

# Ensure project root is on sys.path for `src.*` imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.vision_modeling.vision_models import TinyInceptionV1
from src.vision_interp.receptive_field import receptive_field_mask
from src.utils import set_seed, load_vision_dataset, load_model
from src.vision_interp.visualization_utils import plot_activation_histogram

def get_receptive_fields(model, image_indices, dummy_image, layer_name):
    unique_image_indices = image_indices.unique(dim=0) # [b, 2]
    rf_cache = {}
    for i, j in unique_image_indices:
        mask, mask_info = receptive_field_mask(model, layer_name, i.item(), j.item(), dummy_image)
        rf_cache[(i.item(), j.item())] = (mask, mask_info)
    return rf_cache

def plot_activating_examples(crops, activation_interval_names, n_activating_examples_per_interval, save_path):
    fig = plt.figure(figsize=(n_activating_examples_per_interval * 2, len(crops) * 2))
    gs = GridSpec(len(crops), n_activating_examples_per_interval)
    
    for i, interval in enumerate(crops):
        for j, img in enumerate(interval):
            ax = fig.add_subplot(gs[i, j])
            if hasattr(img, 'ndim') and img.ndim == 4:
                img = rearrange(img, 'b h w c -> h w c')
            ax.imshow(img)
            ax.axis('off')
            if j == 0:
                ax.set_title(activation_interval_names[i])
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="path to the unified config file")
    ap.add_argument("--mode", choices=['feature', 'neuron'], default='feature', 
                   help="mode: feature (SAE) or neuron (model channels)")
    args = ap.parse_args()
    
    # Load the config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Use command line mode, fallback to config mode, default to feature
    mode = args.mode or config.get('mode', 'feature')
    if mode not in ['feature', 'neuron']:
        raise ValueError(f"Mode must be 'feature' or 'neuron', got {mode}")
    
    print(f"Running in {mode} mode")
    
    print(f"Loaded config from: {args.config}")
    
    # Set device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}.")
    
    # Set seed for reproducibility
    set_seed(config.get('seed', 42))

    # Data
    train_dataset, _, num_classes, image_size = load_vision_dataset(config['dataset'], config['batch_size'], config['num_workers'], None, None, shuffle_train=False, return_dataloader=False)
    print(f"Loaded {len(train_dataset)} images.")

    # Model
    config['model_path'] = None
    config['num_classes'] = num_classes
    config['image_size'] = image_size
    model = load_model(config, device)
    model.eval()
    
    # Load activations based on mode
    if mode == 'feature':
        # SAE activations
        sae_activations = torch.load(config['activations_path'])
        all_activations = sae_activations['activations']
        all_dataset_indices = sae_activations['dataset_indices']
        all_position_indices = sae_activations['image_indices']
        print(f"Loaded {all_activations.shape[0]} samples for {all_activations.shape[1]} SAE features.")
    else:  # neuron mode
        # Model activations from the data directory
        from src.utils import load_saved_vision_model_activations
        all_activations, all_dataset_indices, all_position_indices = load_saved_vision_model_activations(
            config['data_dir'], config.get('n_samples_per_image', 128)
        )
        print(f"Loaded {all_activations.shape[0]} samples for {all_activations.shape[1]} model channels.")

    # Cache receptive fields for each potential position in the feature map
    dummy_image = torch.zeros(1, 3, image_size, image_size).to(device)
    rf_cache = get_receptive_fields(model, all_position_indices, dummy_image, config['layer_name'])
    print(f"Created cache for {len(rf_cache)} receptive fields.")

    # Get activating examples for each feature
    activation_intervals = [0, 0.05, 0.3, 0.6, 0.9]
    activation_interval_names = ["0% activation", "5-30% activation", "30-60% activation", "60-90% activation", "90-100% activation"]

    n_activating_examples_per_interval = config.get('examples_per_interval', 4)
    feature_name = "channels" if mode == "neuron" else "features"
    for f in tqdm(range(all_activations.shape[1]), desc=f"Processing {feature_name}"):
        max_activation = all_activations[:, f].max()
        thresholds = [activation_intervals[i] * max_activation for i in range(len(activation_intervals))]
        crops = []
        used_image_indices = []
        
        # Process intervals from highest to lowest
        for i in range(len(activation_intervals) - 1, -1, -1):
            threshold_dataset_indices = []
            shuffle = True
            if i == len(activation_intervals) - 1:
                # For the highest interval (90-100%), use >= 0.9*max_activation (and take the top activations in order)
                relevant_idxs = torch.where(all_activations[:, f] >= thresholds[i])[0]
                shuffle = False
            elif i == 0:
                # For the 0% interval, use exactly 0 activation
                relevant_idxs = torch.where(all_activations[:, f] == 0)[0]
            else:
                # For other intervals, use >= lower_threshold AND < upper_threshold
                relevant_idxs = torch.where((all_activations[:, f] >= thresholds[i]) & (all_activations[:, f] < thresholds[i+1]))[0]
            
            if relevant_idxs.nelement() > 0:
                if shuffle:
                    relevant_idxs = relevant_idxs.view(-1)[torch.randperm(relevant_idxs.shape[0])].view(relevant_idxs.size())
                j = 0
                while j < n_activating_examples_per_interval and j < relevant_idxs.shape[0]:
                    if all_dataset_indices[relevant_idxs[j]] not in used_image_indices:
                        threshold_dataset_indices.append(all_dataset_indices[relevant_idxs[j]])
                        used_image_indices.append(all_dataset_indices[relevant_idxs[j]])
                    j += 1
                relevant_images = train_dataset.data[threshold_dataset_indices]
                relevant_position_indices = all_position_indices[relevant_idxs[:len(threshold_dataset_indices)]]
                interval_crops = []
                for k, (image, position_idx) in enumerate(zip(relevant_images, relevant_position_indices)):
                    mask, mask_info = rf_cache[position_idx[0].item(), position_idx[1].item()]
                    rf_h0, rf_h1, rf_w0, rf_w1 = mask_info['rf_bounds']
                    
                    # Create a crop that shows the full receptive field including padding
                    # Handle cases where RF extends beyond image boundaries
                    img_h, img_w = image.shape[:2]
                    
                    # Check if receptive field encompasses the entire image
                    if rf_h0 <= 0 and rf_h1 >= img_h - 1 and rf_w0 <= 0 and rf_w1 >= img_w - 1:
                        # RF encompasses the entire image, just use the full image
                        crop = image
                    else:
                        # Calculate the crop bounds, treating padding regions as zeros
                        crop_h0 = max(0, rf_h0)
                        crop_h1 = min(img_h - 1, rf_h1)
                        crop_w0 = max(0, rf_w0)
                        crop_w1 = min(img_w - 1, rf_w1)
                        
                        # Create the crop
                        crop = image[crop_h0:crop_h1+1, crop_w0:crop_w1+1]
                        
                        # If the RF extends beyond image boundaries, we need to pad the crop
                        # to show the full receptive field context
                        if rf_h0 < 0 or rf_h1 >= img_h or rf_w0 < 0 or rf_w1 >= img_w:
                            # Calculate padding needed
                            pad_top = max(0, -rf_h0)
                            pad_bottom = max(0, rf_h1 - (img_h - 1))
                            pad_left = max(0, -rf_w0)
                            pad_right = max(0, rf_w1 - (img_w - 1))
                            
                            # Pad the crop with zeros to show the full RF
                            if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                                if len(crop.shape) == 3:  # RGB image
                                    crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                                                mode='constant', constant_values=0)
                                else:  # Grayscale
                                    crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                                                mode='constant', constant_values=0)
                    
                    interval_crops.append(crop)
                crops.append(interval_crops)
            else:
                print(f"No relevant indices found for interval {i} ({activation_interval_names[i]}).")
                crops.append([])  # Add empty list to maintain indexing
                continue

        # Ensure output directory exists
        if mode == 'feature':
            output_dir = os.path.join(config['sae_dir'], 'feature_activating_examples')
        else:  # neuron mode
            output_dir = os.path.join(config['sae_dir'], 'neuron_activating_examples')
        os.makedirs(output_dir, exist_ok=True)
        
        save_path = os.path.join(output_dir, f"{mode}_{f}.png")
        crops.reverse()  # Reverse to match activation_interval_names order
        plot_activating_examples(crops, activation_interval_names, n_activating_examples_per_interval, save_path)

    # Plot activation histogram
    histogram_path = os.path.join(config['sae_dir'], f'{mode}_activation_histogram.png')
    plot_activation_histogram(all_activations, str(histogram_path), mode.capitalize(), f"{mode.capitalize()} Activation Frequencies", 'skyblue' if mode == 'feature' else 'lightcoral')


if __name__ == "__main__":
    main()