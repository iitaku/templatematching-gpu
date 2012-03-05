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

int gpu_module_init(GLuint gl_pbo);
int gpu_module_finish(void);
int gpu_module_compute(Pixel *data, int width, int height, int counter);

#ifdef __cplusplus
}
#endif
#endif /* GPU_MODULE_H */
