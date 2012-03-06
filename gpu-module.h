#ifndef GPU_MODULE_H
#define GPU_MODULE_H
#ifdef __cplusplus
extern "C" {
#endif

#include <opencv2/opencv.hpp>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include "type.h"

int gpu_module_init(GLuint gl_pbo, IplImage *template_image, int width, int height, int interval);
int gpu_module_finish(void);
int gpu_module_compute(IplImage *camera_image);

#ifdef __cplusplus
}
#endif
#endif /* GPU_MODULE_H */
