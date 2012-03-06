#include <iostream>
#include <cassert>

#include <cuda_gl_interop.h>
#include "gpu-module.h"

#define CUDA_CHECK_ERROR() {             \
  cudaError_t err = cudaGetLastError();  \
  if (cudaSuccess != err) {              \
    std::cerr << __FILE__ << ":"         \
              << __LINE__ << ":"         \
              << cudaGetErrorString(err) \
              << std::endl;              \
  }                                      \
  assert(cudaSuccess == err);            \
}

namespace {
  struct cudaGraphicsResource *global_cuda_resource;
}

__constant__ Pixel constant_template_data[32][32];

__global__
void template_matching(Pixel *ptr, int image_width, int image_height)
{
    //const unsigned int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    //const unsigned int yidx = blockIdx.y * blockDim.y + threadIdx.y;
    //ptr[image_width * yidx + xidx] 0;
    return;
}

int gpu_module_init(GLuint gl_pbo, Pixel template_data[32][32])
{
  cudaGLSetGLDevice(0);
  
  cudaGraphicsGLRegisterBuffer(&global_cuda_resource, gl_pbo, cudaGraphicsMapFlagsWriteDiscard);
  CUDA_CHECK_ERROR();

  cudaMemcpyToSymbol(constant_template_data, template_data, 32*32*sizeof(Pixel));
  
  return 0;
}

int gpu_module_finish(void)
{
  cudaGraphicsUnregisterResource(global_cuda_resource);
  return 0;
}

int gpu_module_compute(Pixel *image_data, 
                       int image_width, 
                       int image_height, 
                       Pixel *template_data, 
                       int template_width, 
                       int template_height,
                       int interval)
{
  cudaGraphicsMapResources(1, &global_cuda_resource, 0);
  CUDA_CHECK_ERROR();

  Pixel *ptr;
  size_t bytes;
  cudaGraphicsResourceGetMappedPointer((void**)&ptr, &bytes, global_cuda_resource);
  CUDA_CHECK_ERROR();

  assert(bytes == image_width * image_height * sizeof(Pixel));
  
  cudaMemcpy(ptr, image_data, bytes, cudaMemcpyHostToDevice);
  
  dim3 block_size = dim3(16, 16);
  dim3 grid_size  = dim3(image_width/16, image_height/16);
  template_matching<<<grid_size, block_size>>>(ptr, image_width, image_height);
  CUDA_CHECK_ERROR();

  cudaGraphicsUnmapResources(1, &global_cuda_resource, 0);
  CUDA_CHECK_ERROR();

  return 0;
}

