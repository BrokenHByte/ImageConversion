#!/bin/bash
mkdir build
cmake -B build
cd build
make
./conversion ../test.jpg
xdg-open out.png
