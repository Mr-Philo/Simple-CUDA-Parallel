#include <iostream>  
#include <vector>  
  
void matrix_multiplication(const std::vector<std::vector<float>> &A,  
                           const std::vector<std::vector<float>> &B,  
                           std::vector<std::vector<float>> &C) {  
  int M = A.size();  
  int N = A[0].size();  
  int P = B[0].size();  
  
  for (int i = 0; i < M; i++) {  
    for (int j = 0; j < P; j++) {  
      float sum = 0;  
      for (int k = 0; k < N; k++) {  
        sum += A[i][k] * B[k][j];  
      }  
      C[i][j] = sum;  
    }  
  }  
}  
  
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
  int P = B[0].size();  
  std::vector<std::vector<float>> C(M, std::vector<float>(P, 0));  
  
  // 串行矩阵乘法计算  
  matrix_multiplication(A, B, C);  
  
  // 输出结果矩阵 C  
  for (int i = 0; i < M; i++) {  
    for (int j = 0; j < P; j++) {  
      std::cout << C[i][j] << " ";  
    }  
    std::cout << std::endl;  
  }  
  
  return 0;  
}  
