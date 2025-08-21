import numpy as np
import random
import torch
import os
import glob
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.vision_modeling.vision_models import TinyInceptionV1, TinierInceptionV1
from torchvision.models.vision_transformer import VisionTransformer

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_vision_dataset(dataset_name, batch_size, num_workers, train_transforms=None, test_transforms=None, shuffle_train=False, return_dataloader=True, prefetch_factor=2):
    # set relevant dataset
    if dataset_name == "cifar100":
        dataset = datasets.CIFAR100
        MEAN, STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    elif dataset_name == "cifar10":
        dataset = datasets.CIFAR10
        MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    
    # types of transforms
    PIL_TRANSFORMS = {
        'random_horizontal_flip': lambda p: transforms.RandomHorizontalFlip(p=p),
        'random_crop': lambda config: transforms.RandomCrop(size=config['size'], padding=config['padding']),
        'random_rotation': lambda degrees: transforms.RandomRotation(degrees=degrees),
        'color_jitter': lambda config: transforms.ColorJitter(
            brightness=config.get('brightness', 0.2),
            contrast=config.get('contrast', 0.2),
            saturation=config.get('saturation', 0.2),
            hue=config.get('hue', 0.1)
        )
    }
    
    TENSOR_TRANSFORMS = {
        'normalize': lambda _: transforms.Normalize(MEAN, STD)
    }
    
    def build_transforms(transform_config):
        if transform_config is None:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD)
            ])
        
        # build transforms in correct order
        pil_transforms = []
        tensor_transforms = []
        
        for name, value in transform_config.items():
            if not value:
                continue
                
            if name in PIL_TRANSFORMS:
                pil_transforms.append(PIL_TRANSFORMS[name](value))
            elif name in TENSOR_TRANSFORMS:
                tensor_transforms.append(TENSOR_TRANSFORMS[name](value))
        
        # compose in correct order: PIL transforms -> ToTensor -> Tensor transforms
        transform_list = pil_transforms + [transforms.ToTensor()] + tensor_transforms
        return transforms.Compose(transform_list)
    
    train_dataset = dataset(root='data', train=True, download=True, transform=build_transforms(train_transforms))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor)
    test_dataset = dataset(root='data', train=False, download=True, transform=build_transforms(test_transforms))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor)
    num_classes = len(train_dataset.classes)
    image_size = train_dataset[0][0].shape[1]
    if return_dataloader:
        return train_loader, test_loader, num_classes, image_size
    else:
        return train_dataset, test_dataset, num_classes, image_size

def load_model(model_config, device):
    model_name = model_config['model_name']
    MODEL_BUILDERS = {
        "TinyInceptionV1": lambda cfg: TinyInceptionV1(
            num_classes=int(cfg['num_classes']),
            dropout=float(cfg.get('dropout', 0.3))
        ),
        "TinierInceptionV1": lambda cfg: TinierInceptionV1(
            num_classes=int(cfg['num_classes']),
            dropout=float(cfg.get('dropout', 0.3))
        ),
        "TinyViT": lambda cfg: VisionTransformer(
            num_classes=int(cfg['num_classes']),
            image_size=int(cfg['image_size']),
            patch_size=4, num_layers=10, num_heads=4,
            hidden_dim=32, mlp_dim=128, dropout=0.1
        ),
    }

    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Model {model_name} not supported.")

    model = MODEL_BUILDERS[model_name](model_config)
    if model_config['model_path'] and model_config['model_path'] != "null":
        model.load_state_dict(torch.load(model_config['model_path'], map_location=device))
    model.to(device)
    return model

def load_saved_vision_model_activations(save_dir, n_samples_per_image=None):
    activation_files = glob.glob(os.path.join(f'{save_dir}/batch_*.pt'))
    
    print(f"Loading activations from {len(activation_files)} files.")
    all_activations = []
    all_indices = []
    all_position_indices = []
    
    for file_path in sorted(activation_files):
        saved_data = torch.load(file_path)
        activation_tensor = saved_data['activations']
        indices = saved_data['dataset_indices']
        position_indices = saved_data['image_indices']
        
        if n_samples_per_image is not None:
            # Sample proportionally to the magnitude of the activations
            n_total = activation_tensor.shape[0]
            n_features = activation_tensor.shape[1]
            
            n_samples = min(n_samples_per_image, n_total)
            
            # Calculate mean of each activation vector
            activation_means = torch.mean(activation_tensor, dim=1)
            
            # Create probability distribution proportional to means
            epsilon = 1e-8
            probabilities = activation_means + epsilon
            probabilities = probabilities / probabilities.sum()
            
            # Sample indices based on the probability distribution
            sample_indices = torch.multinomial(probabilities, n_samples, replacement=False)
            activation_sampled = activation_tensor[sample_indices]
            indices_sampled = indices[sample_indices]
            position_indices_sampled = position_indices[sample_indices]
            
            all_activations.append(activation_sampled)
            all_indices.append(indices_sampled)
            all_position_indices.append(position_indices_sampled)
        else:
            # No sampling, use all activations as they are
            all_activations.append(activation_tensor)
            all_indices.append(indices)
            all_position_indices.append(position_indices)
    
    all_activations = torch.cat(all_activations, dim=0) if all_activations else torch.empty(0)
    all_indices = torch.cat(all_indices, dim=0) if all_indices else torch.empty(0)
    all_position_indices = torch.cat(all_position_indices, dim=0) if all_position_indices else torch.empty(0)
    return all_activations, all_indices, all_position_indices