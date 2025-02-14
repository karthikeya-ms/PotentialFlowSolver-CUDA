#!/bin/bash
# Build CPU code
g++ -O3 -fopenmp src/cpu/laplace_cpu.cpp -o laplace_cpu
# Build GPU code (assuming nvcc is in your PATH)
nvcc -O3 src/gpu/laplace_cuda.cu -o laplace_cuda

# Clear previous performance logs
rm -f benchmarks/performance.csv
echo "Version,Time_sec" > benchmarks/performance.csv

# Run CPU simulation
./laplace_cpu

# Run GPU simulation
./laplace_cuda

echo "Benchmarking complete. See benchmarks/performance.csv for results."
