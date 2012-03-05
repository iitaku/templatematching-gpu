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

__global__
void template_matching(Pixel *ptr, int width, int height, int counter)
{
    //const unsigned int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    //const unsigned int yidx = blockIdx.y * blockDim.y + threadIdx.y;
    //ptr[width * yidx + xidx] = make_uchar4(counter % 255, 0, 0, 0);
    return;
}

int gpu_module_init(GLuint gl_pbo)
{
  cudaGLSetGLDevice(0);
  
  cudaGraphicsGLRegisterBuffer(&global_cuda_resource, gl_pbo, cudaGraphicsMapFlagsWriteDiscard);
  CUDA_CHECK_ERROR();

  return 0;
}

int gpu_module_finish(void)
{
  cudaGraphicsUnregisterResource(global_cuda_resource);
  return 0;
}

int gpu_module_compute(Pixel *data, int width, int height, int counter)
{
  cudaGraphicsMapResources(1, &global_cuda_resource, 0);
  CUDA_CHECK_ERROR();

  Pixel *ptr;
  size_t bytes;
  cudaGraphicsResourceGetMappedPointer((void**)&ptr, &bytes, global_cuda_resource);
  CUDA_CHECK_ERROR();

  assert(bytes == width * height * sizeof(Pixel));
  
  cudaMemcpy(ptr, data, bytes, cudaMemcpyHostToDevice);
  
  dim3 block_size = dim3(16, 16);
  dim3 grid_size  = dim3(width/16, height/16);
  template_matching<<<grid_size, block_size>>>(ptr, width, height, counter);
  CUDA_CHECK_ERROR();

  cudaGraphicsUnmapResources(1, &global_cuda_resource, 0);
  CUDA_CHECK_ERROR();

  return 0;
}

