#include <iostream>
#include <math.h>

//kernel function to add two arrays in GPU -> __global__ keyword
__global__
void add(int n, float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("gridDim.x: %d, blockDim.x: %d, blockIdx.x: %d, threadIdx.x: %d, i: %d\n", gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, i);
    if (i < n) {
        y[i] = x[i] + y[i];
    }
}

int main() {
    int N = 1 << 20; // ~1M elements 2^(20)

    // allocate unified memory -> accessible from both CPU and GPU
    float *x, *y;
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // add the arrays on the GPUs
    // <<<number of blocks per grid, number of threads per block>>>
    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    add<<<num_blocks, threads_per_block>>>(N, x, y);
    
    // wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // verify the result -> all values in y should be 3.0f
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }

    std::cout << "Max error: " << maxError << std::endl;

    // free unified memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}