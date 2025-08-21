#!/bin/bash

#SBATCH --output=src/vision_modeling/logs/save_vision_model_activations.out
#SBATCH --exclude=gpu2106
#SBATCH -p 3090-gcondo --gres gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH -J save-vision-model-activations
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

# Run with: sbatch src/vision_modeling/scripts/run_save_vision_model_activations.sh src/vision_modeling/configs/vision_model_activation_saving.yaml

echo "Starting vision model activation saving."

nvidia-smi
source .venv/bin/activate

PYTHONPATH=. uv run src/vision_modeling/save_vision_model_activations.py --config $1
