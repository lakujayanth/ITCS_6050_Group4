#!/bin/bash

# Exit on any error
set -e

# Ensure Python and required dependencies are available
command -v python3 >/dev/null 2>&1 || { echo "Python3 is required but not installed."; exit 1; }

# Install dependencies if not already installed
if [ ! -f "requirements.txt" ]; then
    echo "requirements.txt not found!"
    exit 1
fi
pip install -r requirements.txt

# Define configurations to run (indices correspond to CONFIGS list in main.py)
CONFIG_INDICES=(0 1 2)

# Run experiments for each configuration
for CONFIG_INDEX in "${CONFIG_INDICES[@]}"; do
    echo "Running experiment with configuration index $CONFIG_INDEX"

    # Create unique output directories with timestamp
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="experiments/config_${CONFIG_INDEX}_${TIMESTAMP}"
    mkdir -p "$OUTPUT_DIR/models"
    mkdir -p "$OUTPUT_DIR/videos"
    mkdir -p "$OUTPUT_DIR/checkpoints"
    mkdir -p "$OUTPUT_DIR/plots"
    mkdir -p "$OUTPUT_DIR/logs"

    # Modify main.py to use this config index (temporary copy)
    cp main.py main_temp.py
    sed -i "s/main(CONFIGS\[0\])/main(CONFIGS[$CONFIG_INDEX])/" main_temp.py

    # Run the training
    PYTHONPATH=. python3 main_temp.py

    # Move outputs to the experiment directory
    mv models/* "$OUTPUT_DIR/models/" || true
    mv videos/* "$OUTPUT_DIR/videos/" || true
    mv checkpoints/* "$OUTPUT_DIR/checkpoints/" || true
    mv plots/* "$OUTPUT_DIR/plots/" || true
    mv logs/* "$OUTPUT_DIR/logs/" || true

    # Clean up temporary file
    rm main_temp.py

    echo "Completed experiment with configuration index $CONFIG_INDEX. Outputs saved to $OUTPUT_DIR"
done

echo "All experiments completed!"