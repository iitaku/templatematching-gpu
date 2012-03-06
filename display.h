#ifndef DISPLAY_H
#define DISPLAY_H

#include <iostream>
#include <iomanip>
#include <sstream>

#include <opencv2/opencv.hpp>
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
    Display(int& argc, char* argv[], int image_width, int image_height)
    {
      glutInit(&argc, argv);
      glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);

      glutInitWindowSize(image_width, image_height);

      glutCreateWindow("Real Template Matching");
      glutDisplayFunc(display_callback);
      glutKeyboardFunc(keyboard_callback);

      F::init(image_width, image_height);

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
  
  static int image_width_;
  static int image_height_;
  static int counter_;
  static GLuint gl_pbo_;
  static GLuint gl_tex_;
  static Pixel *image_data_;
  static CvCapture *capture_;

  static void init(int image_width, int image_height)
  {
    image_width_ = image_width;
    image_height_ = image_height;
    image_data_ = new Pixel[image_width_*image_height_];
    
    /* initialize extensions */
    assert(GLEW_OK == glewInit());

    /* pixel buffer */
    glGenBuffers(1, &gl_pbo_);

    GL_CHECK_ERROR();

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_pbo_);
  
    glBufferData(GL_PIXEL_UNPACK_BUFFER, image_width_ * image_height_ * sizeof(Pixel), 0, GL_DYNAMIC_DRAW);

    GL_CHECK_ERROR();
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  
    /* texture */
    glGenTextures(1, &gl_tex_);
    glBindTexture(GL_TEXTURE_2D, gl_tex_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, image_width_, image_height_,  0, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    GL_CHECK_ERROR();
    
    IplImage *template_image = cvLoadImage("template_image.png", CV_LOAD_IMAGE_GRAY);
    
    /* initialize CUDA */
    gpu_module_init(gl_pbo_, );
  }

  static void finish(void)
  {
      delete[] image_data_;
      cvReleaseCapture(&capture_);
  }

  static void compute(void)
  {
    if (NULL == capture_)
    {
        /* initialize OpenCV */
        capture_ = cvCreateCameraCapture(0);
        assert(NULL != capture_);

        cvSetCaptureProperty(capture_, CV_CAP_PROP_FRAME_WIDTH, image_width_);
        cvSetCaptureProperty(capture_, CV_CAP_PROP_FRAME_HEIGHT, image_height_);

        IplImage* image = cvQueryFrame( capture_ );
        assert(3 == image->nChannels);
        assert(8 == image->depth);
    }
    
    IplImage* image = cvQueryFrame( capture_ );

    for(int i=0; i<image->image_height; ++i)
    {
        for(int j=0; j<image->image_width; ++j)
        {
            image_data_[i*image_width_+j] = image->imageData[i*image->image_widthStep+3*j+0];
        }
    }

    gpu_module_compute(image_data_, image_width_, image_height_, counter_);
    counter_++;
  }

  static void display(void)
  {
    glClear(GL_COLOR_BUFFER_BIT);

    glBindTexture(GL_TEXTURE_2D, gl_tex_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_pbo_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width_, image_height_, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);
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

int CaptureAndDrawImage::image_width_ = 0;
int CaptureAndDrawImage::image_height_ = 0;
int CaptureAndDrawImage::counter_ = 0;
GLuint CaptureAndDrawImage::gl_pbo_ = 0;
GLuint CaptureAndDrawImage::gl_tex_ = 0;
Pixel *CaptureAndDrawImage::image_data_ = NULL;
CvCapture *CaptureAndDrawImage::capture_ = NULL;

#endif /* DISPLAY_H */
