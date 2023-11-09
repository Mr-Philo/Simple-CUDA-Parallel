#include <opencv2/opencv.hpp>  
#include <cuda_runtime.h>  
#include <chrono>  
  
using namespace std;
using namespace std::chrono; 

// create Gaussian Kernel
double* createGaussianKernel(int ksize, double sigma) {  
    double* kernel = new double[ksize * ksize];  
    int half_size = ksize / 2;  
    double sum = 0;  
    for(int i = -half_size; i <= half_size; i++) {  
        for(int j = -half_size; j <= half_size; j++) {  
            double value = exp(-(i*i + j*j) / (2*sigma*sigma)) / (2*M_PI*sigma*sigma);  
            kernel[(i + half_size) * ksize + (j + half_size)] = value;  
            sum += value;  
        }  
    }  
    for(int i = 0; i < ksize * ksize; i++) {
        kernel[i] /= sum;
    }
    return kernel;  
}

// CUDA kernel to apply Gaussian filter
__global__ void applyFilterParallel(uchar3* src, uchar3* dst, double* kernel, int rows, int cols, int ksize) {  
    int x = blockIdx.x * blockDim.x + threadIdx.x;  
    int y = blockIdx.y * blockDim.y + threadIdx.y;  
    int half_size = ksize / 2;  
      
    if (x >= half_size && y >= half_size && x < cols - half_size && y < rows - half_size) {  
        double3 sum = make_double3(0, 0, 0);  
  
        for (int i = -half_size; i <= half_size; i++) {  
            for (int j = -half_size; j <= half_size; j++) {  
                uchar3 pixel = src[(y + i) * cols + (x + j)];
                double kernel_val = kernel[(i + half_size) * ksize + (j + half_size)];  
  
                sum.x += pixel.x * kernel_val;  
                sum.y += pixel.y * kernel_val;  
                sum.z += pixel.z * kernel_val;  
            }  
        }  
  
        dst[y * cols + x] = make_uchar3(min(max(int(sum.x), 0), 255), min(max(int(sum.y), 0), 255), min(max(int(sum.z), 0), 255));  
    }  
}

// Sequential version of the Gaussian filter  
cv::Mat applyFilterSequential(const cv::Mat& src, double* kernel, int ksize) {  
    int half_size = ksize / 2;  
    cv::Mat dst = src.clone();
    for(int i = half_size; i < src.rows - half_size; i++) {  
        for(int j = half_size; j < src.cols - half_size; j++) {  
            cv::Vec3f sum(0, 0, 0);  
            for(int k = -half_size; k <= half_size; k++) {  
                for(int l = -half_size; l <= half_size; l++) {  
                    sum += src.at<cv::Vec3b>(i+k, j+l) * kernel[(k + half_size) * ksize + (l + half_size)];  
                }  
            }  
            dst.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(sum[0]);  
            dst.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(sum[1]);  
            dst.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(sum[2]);  
        }  
    }  
    return dst;  
}  
  
// CUDA version of the Gaussian filter  
cv::Mat applyFilterCUDA(const cv::Mat& src, double* kernel, int ksize) {  
    
    // prepare kernel
    double* d_filter; 
    cudaMalloc(&d_filter, ksize * ksize * sizeof(double));
    cudaMemcpy(d_filter, kernel, ksize * ksize * sizeof(double), cudaMemcpyHostToDevice); 

    // prepare image matrix
    uchar3* d_input;  
    uchar3* d_output;  
    cv::Mat dst = src.clone();
  
    cudaMalloc((void**)&d_input, src.rows * src.cols * src.channels());  
    cudaMalloc((void**)&d_output, src.rows * src.cols * src.channels());  
  
    cudaMemcpy(d_input, src.data, src.rows * src.cols * src.channels(), cudaMemcpyHostToDevice);  
    cudaMemcpy(d_output, src.data, src.rows * src.cols * src.channels(), cudaMemcpyHostToDevice);
  
    dim3 block(16, 16);  
    dim3 grid((src.cols + block.x - 1) / block.x, (src.rows + block.y - 1) / block.y);  
    applyFilterParallel<<<grid, block>>>(d_input, d_output, d_filter, src.rows, src.cols, ksize);  
  
    cudaMemcpy(dst.data, d_output, src.rows * src.cols * src.channels(), cudaMemcpyDeviceToHost);  
  
    cudaFree(d_input);  
    cudaFree(d_output); 
    cudaFree(d_filter);

    return dst;
}  
  
bool areEqual(const cv::Mat& a, const cv::Mat& b) {  
    if (a.rows != b.rows || a.cols != b.cols || a.channels() != b.channels())  
        return false;  
    cout << "Total pixels: " << a.rows * a.cols * 3 << "\n";     // debug
    int notEqual = 0;
    for (int i = 0; i < a.rows; i++) {  
        for (int j = 0; j < a.cols; j++) {  
            for (int c = 0; c < a.channels(); c++) {  
                // if ( i == 123 and j == 123) cout << "Pixel value" << (int)a.at<cv::Vec3b>(i, j)[c] << " " << (int)b.at<cv::Vec3b>(i, j)[c] << "\n";     // debug: picel level between 0-255
                // observe that the absolute difference between pixels of the two images is greater than 1, but usually less than 4
                if (abs(a.at<cv::Vec3b>(i, j)[c] - b.at<cv::Vec3b>(i, j)[c]) > 8)       // absolute difference less than 8 (out of 255) is bearable 
                    notEqual++;
            }  
        }  
    }  
    cout << "Not equal pixels: " << notEqual << "\n";        // debug
    double ratio = (double)notEqual / (a.rows * a.cols * 3);      // 3 channel
    if (ratio > 0.01) {
        cout << "Not equal ratio: " << ratio << "\n";       // debug
        return false;
    }
    return true;  
}

int main() {  
    cv::Mat src = cv::imread("puppy_large.jpg");  
    int ksize = 9;
    double sigma = 3.0;
    double* kernel = createGaussianKernel(ksize, sigma);
  
    auto start1 = high_resolution_clock::now();  
    cv::Mat dst1 = applyFilterSequential(src, kernel, ksize);  
    auto end1 = high_resolution_clock::now();  
    duration<double> diff1 = end1 - start1;  
  
    auto start2 = high_resolution_clock::now();  
    cv::Mat dst2 = applyFilterCUDA(src, kernel, ksize);  
    auto end2 = high_resolution_clock::now();  
    duration<double> diff2 = end2 - start2;  
  
    double speedup = diff1.count() / diff2.count();  
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    // cout << deviceCount << endl;     # 8
    double efficiency = speedup / deviceCount;  
  
    cout << "Sequential time: " << diff1.count() << " s\n";  
    cout << "Parallel time: " << diff2.count() << " s\n";  
    cout << "Speedup: " << speedup << "\n";  
    cout << "Efficiency: " << efficiency << "\n";  
    
    if (areEqual(dst1, dst2)) {  
        std::cout << "Results are equal.\n";  
    } else {  
        std::cout << "Results are not equal.\n";  
    }  
    cv::imwrite("compare_seq.jpg", dst1);        // debug
    cv::imwrite("compare_cuda.jpg", dst2);       // debug

    return 0;  
}  

// nvcc -o compare compare.cu `pkg-config --cflags --libs opencv` && ./compare
