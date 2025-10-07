#!/bin/bash -e
#

cd Thirdparty/DBoW2

if [ ! -d build ]; then
  mkdir build
fi
cd build
cmake -DOpenCV_DIR="$(pwd)/../../../../Thirdparty/opencv-3.4.1/build" ..
make -j16
cd ../../..
