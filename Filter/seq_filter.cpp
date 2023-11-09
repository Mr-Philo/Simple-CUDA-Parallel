#include <opencv2/opencv.hpp>  
#include <math.h>  
  
using namespace std;

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
  
// 应用高斯滤波器  
cv::Mat applyFilter(const cv::Mat& src, double* kernel, int ksize) {  
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
  
int main() {  
    cv::Mat src = cv::imread("puppy.jpg");  
    int ksize = 9;
    double sigma = 3.0;
    double* kernel = createGaussianKernel(ksize, sigma);
    // check the kernel value: 
    for(int i=0; i<ksize*ksize; i++){
        cout << kernel[i] << " ";
    }
    cv::Mat dst = applyFilter(src, kernel, ksize);  
    cv::imwrite("output_seq.jpg", dst);  
    return 0;  
}   

// g++ seq_filter.cpp -o seq_filter `pkg-config --cflags --libs opencv` && ./seq_filter
