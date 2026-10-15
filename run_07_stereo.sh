#!/bin/bash
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$PROJECT_ROOT"

# Ensure the conda environment binaries (dsp-slam-yolo) are at the top of PATH
export PATH="/opt/miniconda3/envs/dsp-slam-yolo/bin:$PATH"

VOCAB_FILE="Vocabulary/ORBvoc.bin"
CONFIG_FILE="configs/KITTI04-12.yaml"
DATASET_PATH="/home/ahmad/Datasets/KITTI/07"
MAP_EXPORT_PATH="map/kitti/seq_07_stereo"

rm -rf "$MAP_EXPORT_PATH"
mkdir -p "$MAP_EXPORT_PATH"

echo "Launching KITTI Stereo 07 with DSP-SLAM..."
echo "  Vocabulary: $VOCAB_FILE"
echo "  Config:     $CONFIG_FILE"
echo "  Dataset:    $DATASET_PATH"
echo "  Output:     $MAP_EXPORT_PATH"

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 ./dsp_slam "$VOCAB_FILE" "$CONFIG_FILE" "$DATASET_PATH" "$MAP_EXPORT_PATH"