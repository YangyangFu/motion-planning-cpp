// write a hellow world program to test cuda intallation
#include <stdio.h>
#include <stdlib.h>

__global__ void cuda_hello(){
    printf("Hello World from GPU! %d\n", threadIdx.x*gridDim.x);
}

int main() {
    printf("Hello World from CPU!\n");
    cuda_hello<<<8,8>>>();
    cudaDeviceSynchronize(); 
    return 0;
}