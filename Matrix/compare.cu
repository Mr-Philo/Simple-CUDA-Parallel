#include <iostream>  
#include <vector>  
#include <chrono>  
#include <random>  
#include <cuda_runtime.h>  
  
// 线程块大小和矩阵切片大小  
#define TILE_SIZE 32  
  
// 生成随机矩阵  
std::vector<std::vector<float>> generate_random_matrix(int rows, int cols) {  
  std::random_device rd;  
  std::mt19937 gen(rd());  
  std::uniform_real_distribution<> dis(0, 1);  
  
  std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols));  
  for (int i = 0; i < rows; ++i) {  
    for (int j = 0; j < cols; ++j) {  
      matrix[i][j] = dis(gen);  
    }  
  }  
  
  return matrix;  
}  
  
// 串行矩阵乘法  
std::vector<std::vector<float>> matrix_multiply_serial(const std::vector<std::vector<float>> &A, const std::vector<std::vector<float>> &B) {  
  int M = A.size();  
  int N = A[0].size();  
  int P = B[0].size();  
  std::vector<std::vector<float>> C(M, std::vector<float>(P, 0));  
  
  for (int i = 0; i < M; i++) {  
    for (int j = 0; j < P; j++) {  
      float sum = 0;  
      for (int k = 0; k < N; k++) {  
        sum += A[i][k] * B[k][j];  
      }  
      C[i][j] = sum;  
    }  
  }  
  
  return C;  
}  
  
// 简单并行矩阵乘法内核  
__global__ void simple_matrix_multiplication_kernel(const float *A, const float *B, float *C, int M, int N, int P) {  
  int row = blockIdx.y * blockDim.y + threadIdx.y;  
  int col = blockIdx.x * blockDim.x + threadIdx.x;  
  
  if (row < M && col < P) {  
    float sum = 0;  
    for (int k = 0; k < N; k++) {  
      sum += A[row * N + k] * B[k * P + col];  
    }  
    C[row * P + col] = sum;  
  }  
}  
  
// 优化并行矩阵乘法内核  
__global__ void optimized_matrix_multiplication_kernel(const float *A, const float *B, float *C, int M, int N, int P) {  
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
  
// 将一维数组结果转换为二维矩阵  
std::vector<std::vector<float>> array_to_matrix(const std::vector<float> &array, int rows, int cols) {  
  std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols));  
  for (int i = 0; i < rows; i++) {  
    for (int j = 0; j < cols; j++) {  
      matrix[i][j] = array[i * cols + j];  
    }  
  }  
  return matrix;  
}  
  
// 执行并行矩阵乘法（简单和优化版本）  
std::vector<std::vector<float>> matrix_multiply_parallel(const std::vector<std::vector<float>> &A, const std::vector<std::vector<float>> &B, bool optimized) {  
  int M = A.size();  
  int N = A[0].size();  
  int P = B[0].size();  
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
  dim3 blockSize = optimized ? dim3(TILE_SIZE, TILE_SIZE) : dim3(16, 16);  
  dim3 gridSize((P + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);  
  
  // 调用CUDA内核函数  
  if (optimized) {  
    optimized_matrix_multiplication_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, P);  
  } else {  
    simple_matrix_multiplication_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, P);  
  }  
  
  // 将结果从设备内存拷贝回主机内存  
  cudaMemcpy(C_array.data(), d_C, sizeof(float) * M * P, cudaMemcpyDeviceToHost);  
  
  // 释放设备内存  
  cudaFree(d_A);  
  cudaFree(d_B);  
  cudaFree(d_C);  
  
  // 将一维数组结果转换为二维矩阵  
  return array_to_matrix(C_array, M, P);  
}  
  
int main() {  
  // 初始化矩阵 A 和 B  
  int M = 800;  
  int N = 1000;  
  int P = 600;  
  std::vector<std::vector<float>> A = generate_random_matrix(M, N);  
  std::vector<std::vector<float>> B = generate_random_matrix(N, P);  
  
  // 测量串行版本性能  
  auto start = std::chrono::high_resolution_clock::now();  
  std::vector<std::vector<float>> C_serial = matrix_multiply_serial(A, B);  
  auto end = std::chrono::high_resolution_clock::now();  
  std::chrono::duration<double> serial_duration = end - start;  
  std::cout << "串行版本执行时间: " << serial_duration.count() << " 秒" << std::endl;  
  
  // 测量简单并行版本性能  
  start = std::chrono::high_resolution_clock::now();  
  std::vector<std::vector<float>> C_parallel_simple = matrix_multiply_parallel(A, B, false);  
  end = std::chrono::high_resolution_clock::now();  
  std::chrono::duration<double> parallel_simple_duration = end - start;  
  std::cout << "简单并行版本执行时间: " << parallel_simple_duration.count() << " 秒" << std::endl;  
  
  // 测量优化并行版本性能  
  start = std::chrono::high_resolution_clock::now();  
  std::vector<std::vector<float>> C_parallel_optimized = matrix_multiply_parallel(A, B, true);  
  end = std::chrono::high_resolution_clock::now();  
  std::chrono::duration<double> parallel_optimized_duration = end - start;  
  std::cout << "优化并行版本执行时间: " << parallel_optimized_duration.count() << " 秒" << std::endl;  
  
  return 0;  
}  
