#include <iostream>  
#include <vector>  
#include <cuda_runtime.h>  
  
// 定义线程块大小和矩阵切片大小  
#define TILE_SIZE 32  
  
// CUDA内核函数，计算矩阵乘法  
__global__ void matrix_multiplication_kernel(const float *A, const float *B, float *C, int M, int N, int P) {  
  int row = blockIdx.y * blockDim.y + threadIdx.y;  
  int col = blockIdx.x * blockDim.x + threadIdx.x;  
  
  // 使用共享内存存储矩阵A和矩阵B的子矩阵  
  __shared__ float shared_A[TILE_SIZE][TILE_SIZE];  
  __shared__ float shared_B[TILE_SIZE][TILE_SIZE];  
  
  float sum = 0;  
  
  // 遍历子矩阵，计算矩阵乘法  
  for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {  
    // 将矩阵A和矩阵B的子矩阵加载到共享内存  
    if (row < M && t * TILE_SIZE + threadIdx.x < N) {  
      shared_A[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];  
    } else {  
      shared_A[threadIdx.y][threadIdx.x] = 0;  
    }  
    if (t * TILE_SIZE + threadIdx.y < N && col < P) {  
      shared_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * P + col];  
    } else {  
      shared_B[threadIdx.y][threadIdx.x] = 0;  
    }  
  
    // 同步线程，确保数据加载完成  
    __syncthreads();  
  
    // 计算子矩阵的乘法结果  
    for (int k = 0; k < TILE_SIZE; k++) {  
      sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];  
    }  
  
    // 同步线程，确保计算完成  
    __syncthreads();  
  }  
  
  // 将结果写入矩阵C  
  if (row < M && col < P) {  
    C[row * P + col] = sum;  
  }  
}  
  
// 将二维矩阵vector转换为一维数组  
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
  dim3 blockSize(TILE_SIZE, TILE_SIZE);  
  dim3 gridSize((P + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);  
  
  // 调用CUDA内核函数  
  matrix_multiplication_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, P);  
  
  // 将结果从设备内存拷贝回主机内存  
  cudaMemcpy(C_array.data(), d_C, sizeof(float) * M * P, cudaMemcpyDeviceToHost);  
  
  // 将一维数组结果转换为二维矩阵  
  for (int i = 0; i < M; i++) {  
    for (int j = 0; j < P; j++) {  
      C[i][j] = C_array[i * P + j];  
    }  
  }  
  
  // 输出结果矩阵  
  for (const auto &row : C) {  
    for (float x : row) {  
      std::cout << x << ' ';  
    }  
    std::cout << std::endl;  
  }  
  
  // 释放设备内存  
  cudaFree(d_A);  
  cudaFree(d_B);  
  cudaFree(d_C);  
  
  return 0;  
}  
