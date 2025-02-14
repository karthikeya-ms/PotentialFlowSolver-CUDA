// File: src/gpu/laplace_cuda.cu
#include <iostream>
#include <cmath>
#include <fstream>
#include <chrono>

#define IDX(i, j, Nx) ((j) * (Nx) + (i))

// CUDA kernel: one Jacobi iteration update
__global__ void jacobiKernel(const double* phi, double* phi_new, int Nx, int Ny, int cx, int cy, int r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1) {
        int dx = i - cx, dy = j - cy;
        // Skip updates inside the cylinder (obstacle)
        if (dx * dx + dy * dy <= r * r) {
            phi_new[IDX(i, j, Nx)] = phi[IDX(i, j, Nx)];
        } else {
            phi_new[IDX(i, j, Nx)] = 0.25 * (phi[IDX(i - 1, j, Nx)] +
                                             phi[IDX(i + 1, j, Nx)] +
                                             phi[IDX(i, j - 1, Nx)] +
                                             phi[IDX(i, j + 1, Nx)]);
        }
    }
}

int main() {
    // Grid and simulation parameters
    const int Nx = 1024, Ny = 1024;
    const int iterMax = 1000;  // fixed iterations for GPU demo
    const double U = 1.0;

    const int size = Nx * Ny * sizeof(double);

    // Allocate host arrays and initialize similarly as in the CPU version
    double* h_phi = new double[Nx * Ny];
    double* h_phi_new = new double[Nx * Ny];
    
    // Initialize with uniform flow: phi = U * x on boundaries
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            double value = U * double(i) / (Nx - 1);
            h_phi[IDX(i, j, Nx)] = value;
        }
    }
    
    // Set cylinder (obstacle) in the center
    const int cx = Nx / 2, cy = Ny / 2;
    const int r = Nx / 8;
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            int dx = i - cx, dy = j - cy;
            if (dx * dx + dy * dy <= r * r)
                h_phi[IDX(i, j, Nx)] = 0.0;
        }
    }
    // Copy to h_phi_new
    memcpy(h_phi_new, h_phi, size);

    // Allocate device arrays
    double *d_phi, *d_phi_new;
    cudaMalloc(&d_phi, size);
    cudaMalloc(&d_phi_new, size);

    // Copy host arrays to device
    cudaMemcpy(d_phi, h_phi, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi_new, h_phi_new, size, cudaMemcpyHostToDevice);

    // Configure CUDA grid/block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((Nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Start timing (using std::chrono)
    auto start = std::chrono::high_resolution_clock::now();

    // Iterative Jacobi update loop on GPU
    for (int iter = 0; iter < iterMax; iter++) {
        jacobiKernel<<<numBlocks, threadsPerBlock>>>(d_phi, d_phi_new, Nx, Ny, cx, cy, r);
        cudaDeviceSynchronize();
        // Swap pointers for next iteration
        std::swap(d_phi, d_phi_new);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double execTime = std::chrono::duration<double>(end - start).count();
    std::cout << "GPU (CUDA) Execution time (for " << iterMax << " iterations): " << execTime << " seconds." << std::endl;

    // Copy final result back to host
    cudaMemcpy(h_phi, d_phi, size, cudaMemcpyDeviceToHost);

    // Write result to file for visualization
    std::ofstream out("phi_gpu.dat");
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            out << h_phi[IDX(i, j, Nx)] << " ";
        }
        out << "\n";
    }
    out.close();

    // Log performance data for visualization
    std::ofstream perf("benchmarks/performance.csv", std::ios::app);
    perf << "GPU," << execTime << "\n";
    perf.close();

    // Free memory
    delete[] h_phi;
    delete[] h_phi_new;
    cudaFree(d_phi);
    cudaFree(d_phi_new);
    return 0;
}
