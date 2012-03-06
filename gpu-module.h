#ifndef GPU_MODULE_H
#define GPU_MODULE_H
#ifdef __cplusplus
extern "C" {
#endif

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include "type.h"

int gpu_module_init(GLuint gl_pbo, Pixel template_data[32][32]);
int gpu_module_finish(void);
int gpu_module_compute(Pixel *image_data, 
                       int image_width, 
                       int image_height, 
                       Pixel *template_data, 
                       int template_width, 
                       int template_height,
                       int interval);
#ifdef __cplusplus
}
#endif
#endif /* GPU_MODULE_H */
