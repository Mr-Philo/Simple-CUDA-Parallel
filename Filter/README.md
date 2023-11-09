在串行版本的高斯滤波器中，每个像素依次处理。这意味着在处理第n个像素时，必须等待第n-1个像素的处理完成。这是因为单个CPU核心在任何给定时刻只能执行一个操作。  
   
然而，在`applyFilterKernel`函数中，我们使用CUDA并行处理。CUDA是一种并行计算平台和API模型，它利用图形处理单元（GPU）的强大处理能力来进行高性能计算。在GPU上，可以同时运行很多线程，这些线程可以并行处理数据。在这个函数中，我们为每个像素分配一个线程，这些线程可以同时运行，因此可以并行处理所有的像素。  
   
函数中的`x`和`y`变量计算出当前线程应该处理的像素的坐标。这是通过从线程块索引（`blockIdx`）、线程维度（`blockDim`）和线程索引（`threadIdx`）计算得出的。这种计算方式使得每个线程都有一个唯一的`(x, y)`坐标，因此每个线程都处理一个唯一的像素。  
   
接下来，对于图像的每个通道，计算3x3邻域内像素的平均值（这是一个简单的高斯滤波器的近似）。这一部分的代码和串行版本非常类似，只是现在所有的像素都是并行处理的。  
   
总的来说，`applyFilterKernel`函数的主要区别在于它是在GPU上并行运行的，而不是在CPU上串行运行。这意味着它可以同时处理多个像素，从而大大提高了处理速度。然而，这也意味着你需要理解CUDA编程模型，并能够管理设备内存和线程。

(base) ruizhe@MSRAGPUM07:/Data/sda/ruizhe/C_code/Filter$ nvcc -o compare compare.cu `pkg-config --cflags --libs opencv` && ./compare
Sequential time: 0.685815 s
Parallel time: 0.151082 s
Speedup: 4.53934
Efficiency: 0.567418
Results are equal.
(base) ruizhe@MSRAGPUM07:/Data/sda/ruizhe/C_code/Filter$ nvcc -o compare compare.cu `pkg-config --cflags --libs opencv` && ./compare
Sequential time: 1.80427 s
Parallel time: 0.153774 s
Speedup: 11.7333
Efficiency: 1.46666
Results are equal.

# 11.9 debug

### 发现cuda程序使用了精度比seq低的高斯核函数，导致产生的图片在像素上不完全相同。在高斯核为5×5时，结果上会有2-3个像素的差异；核大小扩充至9×9时，像素差异会更大，有25%的像素都有着至少8的差异

在将数据从主机（CPU）拷贝到设备（GPU）时，cudaMemcpy() 函数本身不会导致精度损失。因为它是一个内存复制操作，不会修改数据的值。然而，如果在创建高斯核或者在处理过程中使用了不同的数据类型，那么可能会有精度损失。

如果高斯核是用 cv::Mat 类型创建的，那么可能需要确认在转换为 double* 类型时没有发生类型转换。比如说，cv::Mat 内部可能使用 float 类型来存储数据，但在转换为 double* 时，数据的值并没有改变，只是类型改变了。在这种情况下，需要确保在创建 cv::Mat 时就使用 double 类型来存储数据。

另一种可能的情况是，cv::Mat 和 CUDA 设备使用了不同的内存对齐方式。CUDA 设备可能需要特定的内存对齐方式来保证最佳性能，而 cv::Mat 可能使用了不同的对齐方式。在这种情况下，可以尝试使用 cudaMallocPitch() 和 cudaMemcpy2D() 来分配和拷贝内存，而不是直接使用 cudaMalloc() 和 cudaMemcpy()。
