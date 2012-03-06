#include <iostream>
#include <cassert>
#include <vector>

#include <opencv2/opencv.hpp>

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
  YUV *global_camera_image_dptr;
  int global_width;
  int global_height;
  int global_interval;
}

__constant__ RGBA constant_template_data[32][32];

__global__
void template_matching(RGBA *frame_buffer, YUV *camera_image, int width, int height)
{
    //const unsigned int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    //const unsigned int yidx = blockIdx.y * blockDim.y + threadIdx.y;
    //ptr[image_width * yidx + xidx] 0;
    return;
}

int gpu_module_init(GLuint gl_pbo, IplImage *template_image, int width, int height, int interval)
{
  global_width = width;
  global_height = height;
  global_interval = interval;

  cudaMalloc(&global_camera_image_dptr, width*height*sizeof(YUV));
  
  cudaGLSetGLDevice(0);
  
  cudaGraphicsGLRegisterBuffer(&global_cuda_resource, gl_pbo, cudaGraphicsMapFlagsWriteDiscard);
  CUDA_CHECK_ERROR();
  
  std::vector<RGBA> template_data(32*32);
  for (int i=0; i<32; ++i)
  {
    for (int j=0; j<32; ++j)
    {
      RGBA pixel;
      pixel.r = template_image->imageData[i*template_image->widthStep+3*j+0];
      pixel.g = template_image->imageData[i*template_image->widthStep+3*j+1];
      pixel.b = template_image->imageData[i*template_image->widthStep+3*j+2];
      pixel.a = 0;
      template_data[i*32+j] = pixel;
    }
  }

  cudaMemcpyToSymbol(constant_template_data, &template_data[0], 32*32*sizeof(RGBA));
  
  return 0;
}

int gpu_module_finish(void)
{
  cudaFree(global_camera_image_dptr);
  
  cudaGraphicsUnregisterResource(global_cuda_resource);
  return 0;
}

int gpu_module_compute(IplImage *camera_image)
{
  cudaMemcpy(global_camera_image_dptr, camera_image->imageData, global_width*global_height*sizeof(RGBA), cudaMemcpyHostToDevice);
  
  cudaGraphicsMapResources(1, &global_cuda_resource, 0);
  CUDA_CHECK_ERROR();

  RGBA *frame_buffer;
  size_t bytes;
  cudaGraphicsResourceGetMappedPointer((void**)&frame_buffer, &bytes, global_cuda_resource);
  CUDA_CHECK_ERROR();

  assert(bytes == global_width * global_height * sizeof(RGBA));
   
  dim3 block_size = dim3(16, 16);
  dim3 grid_size  = dim3(global_width/16, global_height/16);
  template_matching<<<grid_size, block_size>>>(frame_buffer, global_camera_image_dptr, global_width, global_height);
  CUDA_CHECK_ERROR();

  cudaGraphicsUnmapResources(1, &global_cuda_resource, 0);
  CUDA_CHECK_ERROR();

  return 0;
}

