#!/bin/bash

# Activate your conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate wingseek-cuda
export PYTHONNOUSERSITE=1

# Set the base directory
base_dir="path/to/input"
model_path="path/to/model"
annotation_dict="dict.json"
config="predict.ini"

# Loop through subdirectories
for sub_dir in "$base_dir"/*/; do
    echo "Processing $sub_dir"

    # Run predict.py
    python predict.py \
        --input_dir "$sub_dir" \
        --config "$config" \
        --model_path "$model_path" \
        --annotation_dict "$annotation_dict"
done
