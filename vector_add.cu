#include <iostream>  
#include <cuda_runtime.h>  
  
__global__ void vector_add(float* out, float* a, float* b, int n) {  
    int index = threadIdx.x + blockIdx.x * blockDim.x;  
    if (index < n) {  
        out[index] = a[index] + b[index];  
    }  
}  
  
int main() {  
    int n = 100000;  
    float *a, *b, *out;  
  
    cudaMallocManaged(&a, n * sizeof(float));  
    cudaMallocManaged(&b, n * sizeof(float));  
    cudaMallocManaged(&out, n * sizeof(float));

    for (int i = 0; i < n; i++) {  
        a[i] = static_cast<float>(i);  
        b[i] = static_cast<float>(i) * 2;  
    }  
  
    int blockSize = 256;  
    int numBlocks = (n + blockSize - 1) / blockSize;  
    vector_add<<<numBlocks, blockSize>>>(out, a, b, n);  
  
    cudaDeviceSynchronize();  
  
    for (int i = 0; i < 10; i++) {  
        std::cout << "out[" << i << "] = " << out[i] << std::endl;  
    }  
  
    cudaFree(a);  
    cudaFree(b);  
    cudaFree(out);  
  
    return 0;  
}  

