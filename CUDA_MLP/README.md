## Bulid

```sh
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

## Usage

```sh
./mlp -d ../data -l 0.01 -e 10 -h 10
./mlp -d ../data -l 0.01 -e 1 -h 10        // testing
cuda-memcheck ./mlp -d ../data -l 0.01 -e 1 -h 10        // debug
```

---

在CUDA中，你可以在内核函数中使用类和对象，但是有一些限制：  
   
1. 类必须在设备代码中可见。你可以在.cu文件中定义类，或者在.h文件中定义类并在.cu文件中包含这个.h文件。类的定义需要使用 `__host__ __device__` 双关键字，这样类的成员函数既可以在主机代码中使用，也可以在设备代码中使用。  
   
```cpp  
class MyClass {  
public:  
    __host__ __device__ MyClass() { /* 构造函数代码 */ }  
    __host__ __device__ void myMethod() { /* 方法代码 */ }  
};  
```  
   
2. 类的对象可以作为内核函数的参数，但对象必须在主机端创建，并且必须通过 cudaMemcpy 函数复制到设备内存中。你不能在设备代码中使用 new 或 delete 关键字创建或删除对象。  
   
```cpp  
MyClass hostObject;  
MyClass* deviceObject;  
cudaMalloc(&deviceObject, sizeof(MyClass));  
cudaMemcpy(deviceObject, &hostObject, sizeof(MyClass), cudaMemcpyHostToDevice);  
```  
   
3. 如果类有动态分配的成员（例如，指针成员），你需要在设备端分配这些成员的内存，并且在复制对象时需要特别处理这些成员。你不能直接复制主机端的指针成员到设备端，因为主机指针在设备端是无效的。  
   
```cpp  
class MyClass {  
public:  
    __host__ __device__ MyClass(int size) : size(size) { data = new float[size]; }  
    __host__ __device__ ~MyClass() { delete[] data; }  
    __host__ __device__ void myMethod() { /* 方法代码 */ }  
   
private:  
    int size;  
    float* data;  
};  
```  
   
总的来说，你可以在CUDA中使用类，但是需要注意设备代码和主机代码的区别，并且需要正确处理类的内存。

在CUDA编程中，我们通常需要在主机（CPU）和设备（GPU）之间传递数据。因此，我们需要在主机端和设备端都有数据的表示。

在这个例子中，MyClass是一个主机端的类，它用于在主机端存储和操作数据。而DeviceMyClass是一个设备端的结构体，它用于在设备端存储和操作数据。

主要有两个原因需要分别定义这两个类/结构体：
主机和设备的内存模型是不同的。在主机端，你可以使用new和delete来管理动态内存。但在设备端，你需要使用CUDA的内存管理函数（如cudaMalloc和cudaFree）来管理设备内存。因此，你不能直接将一个包含动态内存的主机类传递给设备，你需要创建一个设备结构体，并为其动态内存成员在设备端分配内存。
主机和设备的代码是分开编译的。CUDA使用NVCC编译器来编译CUDA代码，它会将设备代码和主机代码分开编译。因此，你不能在设备代码中直接使用主机类，你需要定义一个设备结构体来在设备代码中使用。

总的来说，MyClass和DeviceMyClass的定义是为了适应主机和设备的不同内存模型和编译模式。同时，这也使得代码更清晰，因为你可以明确地看到哪些数据是在主机端使用的，哪些数据是在设备端使用的。