#include <opencv2/opencv.hpp>  
#include <cuda_runtime.h>  
#include <math.h>
  
using namespace std;

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
__global__ void applyGaussianFilterKernel(uchar3* src, uchar3* dst, double* kernel, int rows, int cols, int ksize) {  
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


// CUDA kernel to apply Mean filter  
__global__ void applyMeanFilterKernel(unsigned char* input, unsigned char* output, int width, int height, int channels)  
{  
    int x = blockIdx.x * blockDim.x + threadIdx.x;  
    int y = blockIdx.y * blockDim.y + threadIdx.y;  
  
    if (x < width && y < height)  
    {  
        for (int c = 0; c < channels; c++)  
        {  
            double sum = 0;  
            for (int i = -1; i <= 1; i++)  
            {  
                for (int j = -1; j <= 1; j++)  
                {  
                    int xIndex = min(max(x + j, 0), width - 1);  
                    int yIndex = min(max(y + i, 0), height - 1);  
                    sum += input[(yIndex * width + xIndex) * channels + c];  
                }  
            }  
            output[(y * width + x) * channels + c] = sum / 9.0f;  
        }  
    }  
}  

  
// Function to apply Gaussian filter using CUDA  
void applyFilter(const cv::Mat& src, cv::Mat& dst)  
{  
    
    // initialize Gaussian kernel
    int ksize = 9;
    double sigma = 3.0;
    double* kernel = createGaussianKernel(ksize, sigma);  
    
    double* d_filter; 
    // check the kernel value:
    for(int i=0; i<ksize*ksize; i++){
        cout << kernel[i] << " ";
    }
    cout << endl;
    cout << "-------------" << endl;
    cudaMalloc(&d_filter, ksize * ksize * sizeof(double));
    cudaMemcpy(d_filter, kernel, ksize * ksize * sizeof(double), cudaMemcpyHostToDevice);

    // debug: check the kernel value
    double* h_filter = new double[ksize * ksize];
    cudaMemcpy(h_filter, d_filter, ksize * ksize * sizeof(double), cudaMemcpyDeviceToHost);
    for(int i=0; i<ksize*ksize; i++){
        cout << h_filter[i] << " ";
    }
    cout << endl;
    delete[] h_filter;

    // initialize input and output image matrix
    uchar3* d_input;  
    uchar3* d_output;  

    cudaMalloc((void**)&d_input, src.rows * src.cols * src.channels());  
    cudaMalloc((void**)&d_output, src.rows * src.cols * src.channels());  
  
    cudaMemcpy(d_input, src.data, src.rows * src.cols * src.channels(), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_output, src.data, src.rows * src.cols * src.channels(), cudaMemcpyHostToDevice);
  
    dim3 block(16, 16);  
    dim3 grid((src.cols + block.x - 1) / block.x, (src.rows + block.y - 1) / block.y);  

    // Using mean filter, kernel size default=3
    // applyMeanFilterKernel<<<grid, block>>>(d_input, d_output, src.cols, src.rows, src.channels());  

    // Using Gaussian filter, need to pass the param 'd_filter'(pre-calculated kernel by fuction 'createGaussianKernel')
    applyGaussianFilterKernel<<<grid, block>>>(d_input, d_output, d_filter, src.rows, src.cols, ksize);
    
    cudaMemcpy(dst.data, d_output, src.rows * src.cols * src.channels(), cudaMemcpyDeviceToHost);  
  
    cudaFree(d_input);  
    cudaFree(d_output);  
    cudaFree(d_filter);
}  
  
int main()  
{  
    cv::Mat src = cv::imread("puppy.jpg");  
    // cv::Mat dst(src.rows, src.cols, src.type());  
    cv::Mat dst = src.clone();
  
    applyFilter(src, dst);  
  
    cv::imwrite("output_parallel.jpg", dst);  
  
    return 0;  
}  

// nvcc -o parallel_filter parallel_filter.cu `pkg-config --cflags --libs opencv` && ./parallel_filter
