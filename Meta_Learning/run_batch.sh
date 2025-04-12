#!/bin/bash

# Create virtual environment (optional)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Ensure ffmpeg is installed (uncomment based on your system)
# sudo apt-get install -y ffmpeg  # Ubuntu/Debian
# brew install ffmpeg  # macOS

# Run experiments with different configs
for config_idx in 0 1 2
do
    echo "Running config $config_idx"
    python main.py --config_idx $config_idx > outputs/config_${config_idx}/log.txt 2>&1
done

# Deactivate virtual environment
deactivate
