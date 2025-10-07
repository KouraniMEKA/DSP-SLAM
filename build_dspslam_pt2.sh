#!/bin/bash -e
#
# This is a build script for DSP-SLAM.
#
conda_base=$(conda info --base)
source "$conda_base/etc/profile.d/conda.sh"
conda activate dsp-slam-pt2

if [ ! -d build ]; then
  mkdir build
fi
cd build
conda_python_bin=`which python`
conda_env_dir="$(dirname "$(dirname "$conda_python_bin")")"
cmake \
  -DOpenCV_DIR="$(pwd)/../../Thirdparty/opencv-3.4.1/build" \
  -DEigen3_DIR="$(pwd)/../../Thirdparty/eigen-3.4.0/install/share/eigen3/cmake" \
  -DPangolin_DIR="$(pwd)/../../Thirdparty/Pangolin-0.9.3/build" \
  -DPYTHON_LIBRARIES="$conda_env_dir/lib/libpython3.10.so" \
  -DPYTHON_INCLUDE_DIRS="$conda_env_dir/include/python3.10" \
  -DPYTHON_EXECUTABLE="$conda_env_dir/bin/python3.10" \
  ..
make -j8

