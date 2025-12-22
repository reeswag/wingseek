#!/bin/bash

# Activate your conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate wingseek-cuda
export PYTHONNOUSERSITE=1

python motion_extraction.py \
    --config motion_extraction.ini \
    --video-dir "/media/will/Data_Analy/WP2/Input" \
    --analysis-dir "/media/will/Data_Analy/WP2/Output" \
    --motion-tuner \
    --mask-tuner