#!/bin/bash

#SBATCH --output=src/vision_interp/logs/sae_pipeline_%j.out
#SBATCH --error=src/vision_interp/logs/sae_pipeline_%j.err
#SBATCH --exclude=gpu2106
#SBATCH -p 3090-gcondo --gres gpu:1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH -J sae-pipeline
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

# SAE Complete Pipeline Script
# Runs training, activation saving, and activating examples generation for a single SAE
# Run with: sbatch src/vision_interp/scripts/sae_run_pipeline.sh src/vision_interp/configs/sae_features7_test.yaml

set -e  # Exit on any error

# Setup Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <config_file> [--training-only]"
    echo "Example: $0 src/vision_interp/configs/sae.yaml"
    echo "Example: $0 src/vision_interp/configs/sae.yaml --training-only"
    exit 1
fi

CONFIG_FILE=$1
TRAINING_ONLY=false

if [ $# -eq 2 ] && [ "$2" = "--training-only" ]; then
    TRAINING_ONLY=true
fi

echo "=========================================="
echo "SAE Pipeline"
echo "Config: $CONFIG_FILE"
echo "Training only: $TRAINING_ONLY"
echo "=========================================="

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Step 1: Train SAE
echo ""
echo "Step 1: Training SAE..."
python src/vision_interp/train_SAE.py --config "$CONFIG_FILE"

# Extract SAE name and models_dir from config after training
CONFIG_VARS=$(python -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
print(f\"{config.get('sae_name', 'auto-generated')} {config.get('models_dir', 'src/vision_interp/models')}\")
")

SAE_NAME=$(echo $CONFIG_VARS | cut -d' ' -f1)
MODELS_DIR=$(echo $CONFIG_VARS | cut -d' ' -f2)
SAE_DIR="$MODELS_DIR/$SAE_NAME"

echo ""
echo "SAE trained and saved to: $SAE_DIR"

if [ "$TRAINING_ONLY" = false ]; then
    # Step 2: Save activations
    echo ""
    echo "Step 2: Saving SAE activations..."
    python src/vision_interp/save_sae_activations.py --sae-dir "$SAE_DIR"

    # Step 3: Generate feature activating examples
    echo ""
    echo "Step 3: Generating feature activating examples..."
    python src/vision_interp/activating_examples.py --config "$SAE_DIR/config.yaml" --mode feature
    
    # Step 4: Generate neuron activating examples
    echo ""
    echo "Step 4: Generating neuron activating examples..."
    python src/vision_interp/activating_examples.py --config "$SAE_DIR/config.yaml" --mode neuron

    # Step 5: Analyze SAE feature branch preferences
    echo ""
    echo "Step 5: Analyzing SAE feature branch preferences..."
    python src/vision_interp/analyze_sae_branches.py --sae-dir "$SAE_DIR"

    echo ""
    echo "=========================================="
    echo "Pipeline completed successfully!"
    echo "Results in: $SAE_DIR"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "Results in: $SAE_DIR"
    echo "=========================================="
fi
