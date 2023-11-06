# Simple-CUDA-Parallel
Simple usage demo of cuda parallel program

## Check the CUDA environment

```sh
nvidia-smi
nvcc --version
```

## Simple usage of nvcc compiler

In the terminal, use the `nvcc` compiler to compile the CUDA program:

```sh
nvcc vector_add.cu -o vector_add  
```

Run the compiled executable: `. /vector_add` and output the following, indicating that the CUDA environment was successfully installed:

```sh
out[0] = 0  
out[1] = 3  
out[2] = 6  
out[3] = 9  
out[4] = 12  
out[5] = 15  
out[6] = 18  
out[7] = 21  
out[8] = 24  
out[9] = 27
```
