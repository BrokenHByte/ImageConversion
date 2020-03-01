
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(image_conversion LANGUAGES CXX CUDA)


find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(conversion conversion.cu)
target_link_libraries( conversion ${OpenCV_LIBS} )
target_compile_features(conversion PUBLIC cxx_std_11)

# /usr/local/share/
# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties( conversion PROPERTIES CUDA_SEPARABLE_COMPILATION ON)




