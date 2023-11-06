#include <iostream>  
#include <vector>  
#include <cuda_runtime.h>  
  
// CUDA内核函数，计算矩阵乘法  
__global__ void matrix_multiplication_kernel(const float *A, const float *B, float *C, int M, int N, int P) {  
  int row = blockIdx.y * blockDim.y + threadIdx.y;  
  int col = blockIdx.x * blockDim.x + threadIdx.x;  
  
  if (row < M && col < P) {     // 防止线程越界
    float sum = 0;  
    for (int k = 0; k < N; k++) {  
      sum += A[row * N + k] * B[k * P + col];  
    }  
    C[row * P + col] = sum;  
  } 
}  

// 将二维矩阵vector转换为一维数组  必须增加这一类型转换，否则会发生报错 Segmentation fault (core dumped)
// 该错误可能是由于在将二维矩阵数据拷贝到设备内存时出现问题。如果使用std::vector<std::vector<float>>来表示二维矩阵，则在将数据拷贝到设备内存时，如果直接使用了A.data()和B.data()，实际上是错误的，因为这两个函数返回的是指向std::vector<float>的指针，而不是指向连续内存的指针。
// 为了解决这个问题，我们需要首先将二维矩阵数据转换为一维数组，然后再将数据拷贝到设备内存。
std::vector<float> matrix_to_array(const std::vector<std::vector<float>> &matrix) {  
  std::vector<float> array;  
  for (const auto &row : matrix) {  
    array.insert(array.end(), row.begin(), row.end());  
  }  
  return array;  
}  

// 主函数  
int main() {  
  // 初始化矩阵 A 和 B  
  std::vector<std::vector<float>> A = {{1, 2, 3},  
                                       {4, 5, 6},  
                                       {7, 8, 9}};  
  std::vector<std::vector<float>> B = {{9, 8, 7},  
                                       {6, 5, 4},  
                                       {3, 2, 1}};  
  
  // 初始化结果矩阵 C  
  int M = A.size();  
  int N = A[0].size();  
  int P = B[0].size();  
  std::vector<std::vector<float>> C(M, std::vector<float>(P, 0));  

  // 将二维矩阵转换为一维数组  
  std::vector<float> A_array = matrix_to_array(A);  
  std::vector<float> B_array = matrix_to_array(B);  
  std::vector<float> C_array(M * P);  

  // 分配设备内存  
  float *d_A, *d_B, *d_C;  
  cudaMalloc((void **)&d_A, sizeof(float) * M * N);  
  cudaMalloc((void **)&d_B, sizeof(float) * N * P);  
  cudaMalloc((void **)&d_C, sizeof(float) * M * P);  
  
  // 拷贝数据到设备内存  
  cudaMemcpy(d_A, A_array.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice);  
  cudaMemcpy(d_B, B_array.data(), sizeof(float) * N * P, cudaMemcpyHostToDevice);  
  
  // 设置线程块大小和网格大小  
  dim3 blockSize(16, 16);  
  dim3 gridSize((P + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);  
  
  // 调用CUDA内核函数  
  matrix_multiplication_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, P);  
  
  // 同步结果  
  cudaMemcpy(C_array.data(), d_C, sizeof(float) * M * P, cudaMemcpyDeviceToHost);  
  
  // 释放设备内存  
  cudaFree(d_A);  
  cudaFree(d_B);  
  cudaFree(d_C);  
  
  // 输出结果矩阵 C  
  for (int i = 0; i < M; i++) {  
    for (int j = 0; j < P; j++) {  
      // std::cout << C[i][j] << " ";  
      std::cout << C_array[i * P + j] << " ";  // C[i][j] = C_array[i * P + j]
    }  
    std::cout << std::endl;  
  }  
  
  return 0;  
}  
