import os
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from src.vision_interp.SAE import VanillaSAE, BatchTopKSAE, TopKSAE
from src.vision_modeling.vision_models import TinyInceptionV1
from src.utils import set_seed


def get_branch_channel_ranges(layer_name, model_name):
    """
    Get the channel ranges for each branch in the specified layer for a given model.
    Channel layout order: [1x1, 3x3, 5x5, pool_proj].
    """
    # Per-model branch channel counts per layer
    MODEL_BRANCH_CHANNELS = {
        'TinyInceptionV1': {
            'stem': [32],  # Single branch for stem (BasicConv2d)
            'features.0': [16, 24, 8, 8],
            'features.1': [16, 24, 8, 8],
            'features.3': [24, 32, 12, 18],
            'features.4': [24, 40, 12, 28],
            'features.6': [32, 48, 16, 40],
            'features.7': [32, 56, 16, 46],
        },
        'TinierInceptionV1': {
            'stem': [22],  # Single branch for stem (BasicConv2d)
            'features.0': [11, 17, 6, 6],
            'features.1': [11, 17, 6, 6],
            'features.3': [17, 23, 8, 13],
            'features.4': [17, 28, 8, 20],
            'features.6': [23, 34, 11, 28],
            'features.7': [23, 40, 11, 33],
        },
    }

    if model_name not in MODEL_BRANCH_CHANNELS:
        raise ValueError(f"Unknown model_name: {model_name}")
    branch_channels = MODEL_BRANCH_CHANNELS[model_name]
    if layer_name not in branch_channels:
        raise ValueError(f"Unknown layer for {model_name}: {layer_name}")

    channels = branch_channels[layer_name]
    ranges = []
    start = 0
    for num_channels in channels:
        end = start + num_channels
        ranges.append((start, end))
        start = end
    return ranges


def load_sae_model(sae_dir):
    """
    Load the SAE model from the saved directory.
    """
    config_path = os.path.join(sae_dir, 'config.yaml')
    weights_path = os.path.join(sae_dir, 'sae.pth')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Calculate activation size from layer name and model
    MODEL_LAYER_CHANNELS = {
        'TinyInceptionV1': {
            'stem': 32,
            'features.0': 56,
            'features.1': 56,
            'features.3': 86,
            'features.4': 104,
            'features.6': 136,
            'features.7': 150,
        },
        'TinierInceptionV1': {
            'stem': 22,
            'features.0': 40,
            'features.1': 40,
            'features.3': 61,
            'features.4': 73,
            'features.6': 96,
            'features.7': 107,
        },
    }

    model_name = config.get('model_name', 'TinyInceptionV1')
    if model_name not in MODEL_LAYER_CHANNELS:
        raise ValueError(f"Unknown model_name: {model_name}")
    per_layer = MODEL_LAYER_CHANNELS[model_name]

    layer_name = config['layer_name']
    if layer_name not in per_layer:
        raise ValueError(f"Unknown layer for {model_name}: {layer_name}")

    # Add activation_size to config
    config['activation_size'] = per_layer[layer_name]
    
    # Create SAE model
    sae_models = {
        "vanilla": VanillaSAE,
        "batch_topk": BatchTopKSAE,
        "topk": TopKSAE
    }
    
    sae_model = sae_models[config['sae_type']](config=config)
    sae_model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    sae_model.eval()
    
    return sae_model, config


def analyze_feature_branch_preferences(sae_model, layer_name, model_name):
    """
    Analyze which branches each SAE feature prefers by examining encoder weights.
    """
    # Get branch channel ranges
    branch_ranges = get_branch_channel_ranges(layer_name, model_name)
    num_branches = len(branch_ranges)
    
    # Get encoder weights (input to hidden layer)
    encoder_weights = sae_model.encoder.weight.data  # Shape: [hidden_size, input_size]
    num_features, input_size = encoder_weights.shape
    
    # Calculate average weight magnitude for each feature per branch
    feature_branch_weights = []
    
    for feature_idx in range(num_features):
        feature_weights = encoder_weights[feature_idx]  # Shape: [input_size]
        
        # Calculate average weight magnitude for each branch
        branch_avgs = []
        for branch_idx, (start_ch, end_ch) in enumerate(branch_ranges):
            branch_weights = feature_weights[start_ch:end_ch]
            avg_magnitude = torch.abs(branch_weights).mean().item()
            branch_avgs.append(avg_magnitude)
        
        # Normalize by the maximum value across branches
        max_val = max(branch_avgs)
        if max_val > 0:
            normalized_avgs = [val / max_val for val in branch_avgs]
        else:
            normalized_avgs = [0.0] * num_branches
        
        feature_branch_weights.append([feature_idx] + normalized_avgs)
    
    return feature_branch_weights, num_branches


def cluster_features_by_branch(feature_branch_weights, threshold=0.5):
    """
    Cluster features based on their branch preferences.
    A feature is assigned to a branch if its normalized weight is above the threshold.
    """
    feature_clusters = []
    
    for feature_data in feature_branch_weights:
        feature_idx = feature_data[0]
        branch_weights = feature_data[1:]
        
        # Find branches where weight is above threshold
        preferred_branches = [i for i, weight in enumerate(branch_weights) if weight >= threshold]
        
        if len(preferred_branches) == 0:
            # If no branch is above threshold, assign to the highest weight branch
            preferred_branches = [np.argmax(branch_weights)]
        elif len(preferred_branches) > 1:
            # If multiple branches are above threshold, keep only the highest
            max_weight = max(branch_weights)
            preferred_branches = [i for i, weight in enumerate(branch_weights) if weight == max_weight]
        
        cluster_data = {
            'feature_idx': feature_idx,
            'preferred_branch': preferred_branches[0]
        }
        # Add individual branch weights
        for i, weight in enumerate(branch_weights):
            cluster_data[f'branch_{i+1}_weight'] = weight
        
        feature_clusters.append(cluster_data)
    
    return feature_clusters


def save_results(sae_dir, feature_branch_weights, feature_clusters, num_branches, layer_name):
    """
    Save analysis results to CSV files.
    """
    # Create DataFrame for feature clusters
    df_clusters = pd.DataFrame(feature_clusters)
    
    # Save to CSV file
    clusters_path = os.path.join(sae_dir, 'feature_branch_clusters.csv')
    df_clusters.to_csv(clusters_path, index=False)
    
    print(f"Saved feature clusters to: {clusters_path}")
    
    # Print summary statistics
    print(f"\nBranch Analysis Summary for {layer_name}:")
    print(f"Total features: {len(feature_branch_weights)}")
    print(f"Number of branches: {num_branches}")
    
    # Count features per branch
    branch_counts = {}
    for cluster in feature_clusters:
        branch = cluster['preferred_branch']
        branch_counts[branch] = branch_counts.get(branch, 0) + 1
    
    print("\nFeatures per branch:")
    for branch in range(num_branches):
        count = branch_counts.get(branch, 0)
        percentage = (count / len(feature_clusters)) * 100
        print(f"  Branch {branch + 1}: {count} features ({percentage:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-dir", required=True, help="path to the SAE directory")
    ap.add_argument("--threshold", type=float, default=0.5, 
                   help="threshold for branch preference (default: 0.5)")
    args = ap.parse_args()
    
    print(f"Analyzing SAE branches for: {args.sae_dir}")
    print(f"Branch preference threshold: {args.threshold}")
    
    # Load SAE model and config
    sae_model, config = load_sae_model(args.sae_dir)
    layer_name = config['layer_name']
    model_name = config.get('model_name', 'TinyInceptionV1')
    
    print(f"Layer: {layer_name}")
    print(f"SAE type: {config['sae_type']}")
    print(f"Expansion factor: {config['expansion_factor']}")
    print(f"TopK: {config['topk']}")
    
    # Analyze feature branch preferences
    feature_branch_weights, num_branches = analyze_feature_branch_preferences(sae_model, layer_name, model_name)
    
    # Cluster features by branch
    feature_clusters = cluster_features_by_branch(feature_branch_weights, args.threshold)
    
    # Save results
    save_results(args.sae_dir, feature_branch_weights, feature_clusters, num_branches, layer_name)
    
    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()
