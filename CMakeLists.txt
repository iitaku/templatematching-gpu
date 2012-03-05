project(templatematching)
cmake_minimum_required(VERSION 2.8)

find_package(CUDA)
find_package(OpenCV)
find_package(OpenGL)
find_package(GLUT)

#set(CUDA_NVCC_FLAGS -arch=sm_20)
cuda_add_library(gpu-module gpu-module.cu)

add_executable(templatematching main.cc)
target_link_libraries(templatematching gpu-module opencv_highgui GLEW ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
