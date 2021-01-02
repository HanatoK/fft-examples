#!/bin/sh
/usr/local/cuda/bin/nvcc -ccbin /usr/bin/g++-7 ./main.cu -lfftw3 -lcufft -O2 -o test
