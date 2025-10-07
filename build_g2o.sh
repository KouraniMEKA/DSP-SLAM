#!/bin/bash -e
#

cd "Thirdparty/g2o"

if [ ! -d build ]; then
  mkdir build
fi
cd build
cmake -DEigen3_DIR="$(pwd)/../../../../Thirdparty/eigen-3.4.0/install/share/eigen3/cmake" ..
make -j8
cd ../..

