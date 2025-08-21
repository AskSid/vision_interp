#!/bin/bash

#SBATCH --output=src/vision_modeling/logs/train_vision_model.out
#SBATCH --exclude=gpu2106
#SBATCH -p gpu --gres gpu:1
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH -J train-vision-model
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

# Run with: sbatch src/vision_modeling/scripts/run_train_vision_model.sh src/vision_modeling/configs/vision_model_training.yaml

echo "Starting vision model training."

nvidia-smi
source .venv/bin/activate

PYTHONPATH=. uv run src/vision_modeling/train_vision_model.py --config $1