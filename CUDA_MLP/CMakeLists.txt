cmake_minimum_required(VERSION 3.8)  
project(mlp)  
  
set(CMAKE_CXX_STANDARD 11)  
set(CMAKE_CXX_STANDARD_REQUIRED ON)  
  
# Enable CUDA  
enable_language(CUDA)  
  
# Add CUDA files  
file(GLOB CUDA_FILES "src/*.cu")  
  
include_directories(include)  
  
# Add source files  
aux_source_directory(src SRC)  
  
# Add CUDA files to the source files  
list(APPEND SRC ${CUDA_FILES})  
  
add_executable(mlp ${SRC})  
  
# Specify the CUDA standard  
set_property(TARGET mlp PROPERTY CUDA_STANDARD 11)  
