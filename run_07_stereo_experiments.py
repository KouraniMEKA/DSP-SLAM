#!/usr/bin/env python3
import argparse
import os
import sys
import time
import subprocess
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def run_experiment(run_idx, num_runs=5, sequence_dir=None, vocab_path=None, config_path=None, binary_path=None):
    """Runs a single run of KITTI stereo sequence 07 with DSP-SLAM."""
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"seq_07_stereo_DSP_SLAM_{dt_str}"
    output_dir = os.path.join(PROJECT_ROOT, "map", "kitti", dir_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n==================================================")
    print(f"Starting Run {run_idx}/{num_runs} for KITTI Stereo 07 (DSP-SLAM)")
    print(f"Output directory: {output_dir}")
    print(f"==================================================")
    
    start_time = time.time()
    try:
        env = os.environ.copy()
        
        # Ensure python environment binaries are available if present
        python_dir = "/opt/miniconda3/envs/dsp-slam-yolo/bin"
        if not os.path.exists(python_dir):
            python_dir = os.path.dirname(sys.executable)
        env["PATH"] = python_dir + os.path.pathsep + env.get("PATH", "")
        if os.path.exists("/usr/lib/x86_64-linux-gnu/libstdc++.so.6"):
            env["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

        cmd = [
            binary_path,
            vocab_path,
            config_path,
            sequence_dir,
            output_dir
        ]
        
        print(f"Executing command: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[Error] Run {run_idx} failed with exit code {e.returncode}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[Error] Run {run_idx} encountered an error: {e}", file=sys.stderr)
        return None

    duration = time.time() - start_time
    print(f"Run {run_idx} finished in {duration:.1f} seconds.")
    print(f"[Success] Output saved to {output_dir}")
    return output_dir

def main():
    default_seq_dir = os.path.expanduser("~/Datasets/KITTI/07")
    if not os.path.exists(default_seq_dir):
        default_seq_dir = os.path.join(PROJECT_ROOT, "data/kitti/07")

    parser = argparse.ArgumentParser(description="KITTI Stereo 07 DSP-SLAM Experiment Runner")
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of runs (default: 5)"
    )
    parser.add_argument(
        "--sequence_dir",
        type=str,
        default=default_seq_dir,
        help=f"Path to KITTI sequence 07 directory (default: {default_seq_dir})"
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default=os.path.join(PROJECT_ROOT, "Vocabulary/ORBvoc.bin"),
        help="Path to ORB vocabulary file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "configs/KITTI04-12.yaml"),
        help="Path to YAML settings file"
    )
    parser.add_argument(
        "--binary",
        type=str,
        default=os.path.join(PROJECT_ROOT, "dsp_slam"),
        help="Path to dsp_slam executable"
    )

    args = parser.parse_args()

    completed_dirs = []
    print(f"\n>>> Running KITTI Stereo 07 with DSP-SLAM ({args.runs} runs)")

    for i in range(1, args.runs + 1):
        dir_path = run_experiment(
            run_idx=i,
            num_runs=args.runs,
            sequence_dir=args.sequence_dir,
            vocab_path=args.vocab,
            config_path=args.config,
            binary_path=args.binary
        )
        if dir_path:
            completed_dirs.append(dir_path)

    print("\n==================================================")
    print("Experiment Summary")
    print("==================================================")
    print(f"Total successful runs: {len(completed_dirs)}/{args.runs}")
    for idx, d in enumerate(completed_dirs, 1):
        print(f"  Run {idx}: {d}")

if __name__ == "__main__":
    main()
