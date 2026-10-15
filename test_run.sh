#!/bin/bash
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$PROJECT_ROOT"

# Ensure the conda environment binaries (dsp-slam-yolo) are at the top of PATH
export PATH="/opt/miniconda3/envs/dsp-slam-yolo/bin:$PATH"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# ------------------------------------------------------------------------------------------
# KITTI stereo example, single frame reconstruction:
Config_file='./configs/config_kitti.json'
pathDataset="$HOME/Datasets/KITTI/07" 

echo "Running single-frame reconstruction test on KITTI 07 (Frame 100)..."
python3 reconstruct_frame.py \
  --config "$Config_file" \
  --sequence_dir "$pathDataset" \
  --frame_id 100
