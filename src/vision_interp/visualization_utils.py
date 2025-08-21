import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_activation_histogram(activations, save_path, activation_type="Feature", title=None, color='skyblue'):
    activations_np = activations.numpy()
    
    if activation_type == "Feature":
        # Count activations as any time above zero
        activation_counts = np.sum(activations_np != 0, axis=0)
    elif activation_type == "Neuron":
        # For neurons, calculate mean activation per neuron and count as activating when > mean + 1 std
        neuron_means = np.mean(activations_np, axis=0)
        neuron_stds = np.std(activations_np, axis=0)
        activation_thresholds = neuron_means + neuron_stds
        activation_counts = np.sum(activations_np > activation_thresholds, axis=0)
    else:
        raise ValueError(f"activation_type must be 'Feature' or 'Neuron', got {activation_type}")
    
    n_samples = activations_np.shape[0]
    
    plt.figure(figsize=(12, 8))
    plt.hist(activation_counts, bins=10, alpha=0.7, color=color, edgecolor='black')
    plt.xlabel(f'{activation_type} Activation Counts (raw number of samples)')
    plt.ylabel(f'Number of {activation_type}s')
    
    if title is None:
        title = f"{activation_type} Activation Counts"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    mean_activation = np.mean(activation_counts)
    median_activation = np.median(activation_counts)
    max_activation = np.max(activation_counts)
    min_activation = np.min(activation_counts)
    
    stats_text = f'Total number of activations: {activations_np.shape[0]}'
    plt.text(0.7, 0.95, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {activation_type.lower()} activation histogram to {save_path}")
    print(f"{activation_type} activation statistics:")
    print(f"  Mean activation count: {mean_activation:.1f}")
    print(f"  Median activation count: {median_activation:.1f}")
    print(f"  Max activation count: {max_activation:.1f}")
    print(f"  Min activation count: {min_activation:.1f}")
