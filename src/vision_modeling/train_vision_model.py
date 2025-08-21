import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import os
import yaml
from torchvision import datasets, transforms
from src.vision_modeling.vision_models import TinyInceptionV1, TinierInceptionV1
import wandb
from tqdm import tqdm
from src.utils import set_seed, load_vision_dataset, load_model

def train(model, train_loader, optimizer, lr_scheduler, criterion, device):
    model.train()
    train_loss = 0
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        lr_scheduler.step()
    return train_loss / len(train_loader)

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss, correct, total, top_5_correct = 0, 0, 0, 0
    with torch.no_grad():
        for _, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            top_5_correct += (torch.topk(output, k=5, dim=1)[1] == target.unsqueeze(1)).sum().item()
    test_loss /= len(test_loader)
    accuracy = 100 * correct / total
    top_5_accuracy = 100 * top_5_correct / total
    return test_loss, accuracy, top_5_accuracy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="path to the config file")
    args = ap.parse_args()
    
    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['seed'])

    if config['wandb']:
        wandb.init(project=f"{config['dataset']}-{config['model_name']}", config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}.")

    prefetch_factor = config.get('prefetch_factor', 2)
    train_loader, test_loader, num_classes, image_size = load_vision_dataset(config['dataset'], int(config['batch_size']), int(config['num_workers']), config['transforms']['train'], config['transforms']['test'], shuffle_train=True, prefetch_factor=prefetch_factor)
    config['num_classes'] = num_classes
    config['image_size'] = image_size
    print(f"Loaded dataset.")
    model = load_model(model_config=config, device=device)
    print(f"Loaded {config['model_name']} model with {sum(p.numel() for p in model.parameters())} parameters.")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['lr']), weight_decay=float(config['weight_decay']))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config['epochs']), eta_min=float(config['lr']) * 0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config['label_smoothing']))

    best_loss = float("inf")

    print(f"Training {config['model_name']} on {config['dataset']} for {config['epochs']} epochs.")
    for epoch in tqdm(range(int(config['epochs'])), desc="Training epochs"):
        train_loss = train(model, train_loader, optimizer, lr_scheduler, criterion, device)
        test_loss, test_accuracy, test_top_5_accuracy = test(model, test_loader, criterion, device)
        
        if test_loss < best_loss:
            best_loss = test_loss
            os.makedirs(f"{config['save_dir']}/{config['dataset']}", exist_ok=True)
            name_addon = config.get('name_addon', '')
            model_filename = f"best_{config['dataset']}_{config['model_name']}{name_addon}.pth"
            torch.save(model.state_dict(), f"{config['save_dir']}/{config['dataset']}/{model_filename}")

        if config['wandb']:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss, 
                "test_loss": test_loss, 
                "test_accuracy": test_accuracy,
                "test_top_5_accuracy": test_top_5_accuracy
            })
    print(f"Training completed!")

if __name__ == "__main__":
    main()