
python reconstruct_frame.py --config configs/config_kitti.json --sequence_dir ~/Datasets/DSP-SLAM/kitti/07 --frame_id 100
python reconstruct_frame.py --config configs/config_kitti_pt2.json --sequence_dir ~/Datasets/DSP-SLAM/kitti/07 --frame_id 100


./dsp_slam Vocabulary/ORBvoc.bin configs/KITTI04-12.yaml ~/Datasets/KITTI/07 map/kitti/07

# ASUS:
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 ./dsp_slam Vocabulary/ORBvoc.bin configs/KITTI04-12-pt2.yaml ~/Datasets/KITTI/07 map/kitti/07
