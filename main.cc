#include <iostream>
#include <cstdio>

#include "display.h"

int main(int argc, char *argv[])
{
  int width = 320;
  int height = 240;

  if (1<argc)
  {
    for (char** ptr = &argv[1]; NULL != *ptr; ++ptr)
    {
      if (0 == strcmp(*ptr, "-geometry"))
      {
        sscanf(*(ptr+1), "%dx%d", &width, &height);
        break;
      }
    }
  }

  if (0 != (width % 16) || 0 != (height % 16))
  {
    std::cerr << "window size must be multiples of 16" << std::endl;
    return -1;
  }

  Display<CaptureAndDrawImage> display(argc, argv, width, height);
  display.loop();

  display.finish();

  return 0;
}
