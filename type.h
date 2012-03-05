#ifndef TYPE_H
#define TYPE_H

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

struct Pixel
{
  GLbyte r;
  GLbyte g;
  GLbyte b;
  GLbyte a;
};

#endif /* TYPE_H */
