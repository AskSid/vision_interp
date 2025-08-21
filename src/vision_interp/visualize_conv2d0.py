import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from vision_modeling.vision_models import TinyInceptionV1, TinierInceptionV1

model = TinierInceptionV1(num_classes=100)
model.load_state_dict(torch.load('src/vision_modeling/models/cifar100/best_cifar100_TinierInceptionV1_tinier_aug.pth', map_location=torch.device('cpu')))
model.eval()

def plot_filters(name, weight, cols=8):
    def normalize_filter(filter):
        min_val = filter.min()
        max_val = filter.max()
        if max_val - min_val == 0:
            normalized_x = torch.zeros_like(filter)
        else:
            normalized_x = (filter - min_val) / (max_val - min_val)
        return normalized_x
    
    num_filters = weight.shape[0]
    rows = math.ceil(num_filters / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(f'{name} Filters', fontsize=24)

    for i in range(num_filters):
        row = i // cols
        col = i % cols
        
        filter = normalize_filter(weight[i])
        axes[row, col].imshow(filter.squeeze(0).permute(1, 2, 0))
        axes[row, col].axis('off')

    for i in range(num_filters, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(f'src/vision_interp/models/TinierInceptionV1/tinier_stem_topk/{name}_filters.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_filters('stem', model.stem.conv.weight.data)