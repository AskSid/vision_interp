#!/bin/bash

#SBATCH --output=src/vision_interp/logs/sae_sweep_%j.out
#SBATCH --error=src/vision_interp/logs/sae_sweep_%j.err
#SBATCH --exclude=gpu2106
#SBATCH -p 3090-gcondo --gres gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -J sae-sweep
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

# SAE Sweep Pipeline Script
# Runs training for a sweep, optionally runs post-processing for each SAE

set -e  # Exit on any error

# Setup Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <sweep_config_file> [--post-process]"
    echo "Example: $0 src/vision_interp/configs/sae_sweep.yaml"
    echo "Example: $0 src/vision_interp/configs/sae_sweep.yaml --post-process"
    exit 1
fi

CONFIG_FILE=$1
POST_PROCESS=false

if [ $# -eq 2 ] && [ "$2" = "--post-process" ]; then
    POST_PROCESS=true
fi

echo "=========================================="
echo "SAE Sweep Pipeline"
echo "Config: $CONFIG_FILE"
echo "Post-process: $POST_PROCESS"
echo "=========================================="

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Step 1: Run sweep training
echo ""
echo "Step 1: Running sweep training..."
python src/vision_interp/train_SAE.py --config "$CONFIG_FILE"

# Extract experiment name from config
EXPERIMENT_NAME=$(python -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
print(config.get('experiment_name', 'unknown'))
")

SWEEP_DIR="src/vision_interp/models/$EXPERIMENT_NAME"

echo ""
echo "Sweep completed! Results in: $SWEEP_DIR"

# Step 2: Post-process each SAE if requested
if [ "$POST_PROCESS" = true ]; then
    echo ""
    echo "Step 2: Running post-processing for each SAE..."
    
    # Find all SAE directories in the sweep
    for sae_dir in "$SWEEP_DIR"/*/; do
        if [ -d "$sae_dir" ] && [ -f "$sae_dir/config.yaml" ]; then
            echo ""
            echo "Processing: $(basename "$sae_dir")"
            
            # Save activations
            echo "  Saving activations..."
            python src/vision_interp/save_sae_activations.py --sae-dir "$sae_dir"
            
            # Generate activating examples
            echo "  Generating activating examples..."
            python src/vision_interp/activating_examples.py --sae-dir "$sae_dir"
        fi
    done
    
    echo ""
    echo "Post-processing completed for all SAEs!"
fi

echo ""
echo "=========================================="
echo "Sweep pipeline completed!"
echo "Results in: $SWEEP_DIR"
echo "CSV results: $SWEEP_DIR/$EXPERIMENT_NAME.csv"
echo "=========================================="
