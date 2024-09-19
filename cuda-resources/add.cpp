#include <iostream>
#include <math.h>

//function to add two arrays in host
void add(int n, float* x, float* y) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}

int main() {
    int N = 1 << 20; // ~1M elements 2^(20)

    // allocate dynamic memory (heap) for arrays
    // without keyword 'new' the memory is allocated on the stack. 
    // stack memory has a limited size and might cause stack overflow if the size is too large
    float* x = new float[N];
    float* y = new float[N];

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // add the arrays on the host: CPU
    add(N, x, y);

    // verify the result -> all values in y should be 3.0f
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }

    std::cout << "Max error: " << maxError << std::endl;

    // free array memory: delete[]
    // if don't delete the memory, it will cause 
    delete[] x;
    delete[] y;

    return 0;
}