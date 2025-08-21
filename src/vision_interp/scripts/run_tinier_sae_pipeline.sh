#!/bin/bash

echo "Starting TinierInceptionV1 SAE pipeline runs..."

CONFIG_DIR="src/vision_interp/configs/TinierInceptionV1"
SCRIPT_PATH="src/vision_interp/scripts/sae_run_pipeline.sh"

configs=(
    "sae_tinier_stem.yaml"
    "sae_tinier_features0.yaml"
    "sae_tinier_features1.yaml"
    "sae_tinier_features3.yaml"
    "sae_tinier_features4.yaml"
    "sae_tinier_features6.yaml"
    "sae_tinier_features7.yaml"
)

for config in "${configs[@]}"; do
    config_path="$CONFIG_DIR/$config"
    echo "Submitting job for $config..."
    sbatch "$SCRIPT_PATH" "$config_path"
done

echo "All TinierInceptionV1 SAE pipeline jobs submitted!"
