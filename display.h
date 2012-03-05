#ifndef DISPLAY_H
#define DISPLAY_H

#include <iostream>
#include <iomanip>
#include <sstream>

#include <opencv/highgui.h>
#include <cuda.h>

#include <GL/glew.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include "type.h"
#include "gpu-module.h"
#include "performance.h"

#define GL_CHECK_ERROR() {           \
  GLenum err = glGetError();         \
  if (GL_NO_ERROR != err) {          \
    std::cerr << __FILE__ << ":"     \
              << __LINE__ << ":"     \
              << gluErrorString(err) \
              << std::endl;          \
  }                                  \
  assert(GL_NO_ERROR == err);        \
}

template<typename F>
class Display
{
  private:

    static void display_callback(void)
    {
      Performance perf("all");

      F::compute();
      F::display();

      perf.stop();

      double fps = 1e3f / perf.mean_ms();
      std::stringstream ss;
      ss << "Realtime Template Matching : "
         << std::setw(5) << std::left << std::setprecision(4) 
         << fps << " fps";

      glutSetWindowTitle(ss.str().c_str());
    }

    static void keyboard_callback(unsigned char key , int x , int y)
    {
      F::keyboard(key, x, y);
    }

  public:
    Display(int& argc, char* argv[], int width, int height)
    {
      glutInit(&argc, argv);
      glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);

      glutInitWindowSize(width, height);

      glutCreateWindow("Real Template Matching");
      glutDisplayFunc(display_callback);
      glutKeyboardFunc(keyboard_callback);

      F::init(width, height);

      GL_CHECK_ERROR();
    }

    void loop()
    {
      glutMainLoop();
    }

    void finish()
    {
      F::finish();
    }

};

struct CaptureAndDrawImage
{
  
  static int width_;
  static int height_;
  static int counter_;
  static GLuint gl_pbo_;
  static GLuint gl_tex_;

  static void init(int width, int height)
  {
    width_ = width;
    height_ = height;
    
    /* initialize extensions */
    assert(GLEW_OK == glewInit());

    /* pixel buffer */
    glGenBuffers(1, &gl_pbo_);

    GL_CHECK_ERROR();

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_pbo_);
  
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4 * sizeof(GLubyte), 0, GL_DYNAMIC_DRAW);

    GL_CHECK_ERROR();
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  
    /* texture */
    glGenTextures(1, &gl_tex_);
    glBindTexture(GL_TEXTURE_2D, gl_tex_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width_, height_,  0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    GL_CHECK_ERROR();

    gpu_module_init(gl_pbo_);
  }

  static void finish(void)
  {
  }

  static void compute(void)
  {
    std::vector<Pixel> data(width_*height_);
    for(int i=0; i<width_*height_; ++i)
    {
      data[i].r = rand();
    }
    gpu_module_compute(&data[0], width_, height_, counter_);
    counter_++;
  }

  static void display(void)
  {
    glClear(GL_COLOR_BUFFER_BIT);

    glBindTexture(GL_TEXTURE_2D, gl_tex_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_pbo_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        
    glBegin(GL_QUADS);
    glVertex2f(-1.0, -1.0); glTexCoord2f(0, 0);
    glVertex2f(-1.0, +1.0); glTexCoord2f(1, 0);
    glVertex2f(+1.0, +1.0); glTexCoord2f(1, 1);
    glVertex2f(+1.0, -1.0); glTexCoord2f(0, 1);
    glEnd();
    glBindTexture(GL_TEXTURE_2D, 0);
    
    glutSwapBuffers();
    
    glutPostRedisplay();
  }

  static void keyboard(unsigned char key, int x, int y)
  {
    switch(key)
    {
      case 'q':
        finish();
        exit(0);
        break;

      default:
        break;
    }
  }
};

int CaptureAndDrawImage::width_ = 0;
int CaptureAndDrawImage::height_ = 0;
int CaptureAndDrawImage::counter_ = 0;
GLuint CaptureAndDrawImage::gl_pbo_ = 0;
GLuint CaptureAndDrawImage::gl_tex_ = 0;

#endif /* DISPLAY_H */
