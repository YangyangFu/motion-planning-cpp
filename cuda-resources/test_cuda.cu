// write a hellow world program to test cuda intallation
#include <stdio.h>
#include <stdlib.h>

// c++ kernel function for cuda: executed N times in parallel by N different CUDA threads
__global__ void cuda_hello(){
    printf("Hellow World from GPU at block (%d, %d, %d), thread (%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}


// c++ kernel to perform parallel addition of two matrices of shape N x N
const int N = 100000;
__global__ void matAdd(float* a[N][N], float* b[N][N], float* c[N][N]) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        c[i][j] = a[i][j] + b[i][j];
}

int main() {
    
    // <<<number of blocks per grid, number of threads per block>>>
    cuda_hello<<<8,8>>>();
    cudaDeviceSynchronize();

    return 0;
}