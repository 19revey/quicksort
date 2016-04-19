// OpenGL Graphics includes                                                     
/*#include <GL/glew.h>
#if defined (WIN32)
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif*/

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<string.h>
#include"particle.h"
//#include"find_neighbor.cpp"
//#include"cudasort.cuh"
#include<cuda_runtime.h>
#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h           
#include <helper_cuda_gl.h> // i

#define CEILING(X) (X-(int)(X) > 0 ? (int)(X+1) : (int)(X))
#define delta_fun(r)  (fabs(r) <= 2 ? ((1. + cos(M_PI*(r)/2.)) * 0.25) : 0.0)
#define min(x,y)  (x<y? x:y)
#define max(x,y)  (x>y? x:y)




int main()
{
    double timeused[3];
    clock_t start = clock ();


  //sph_param param;
  sph_fluid s;
  s.place_particles();
  s.normalize_mass();
  s.print_info();


  s.iterations(timeused);

        clock_t timeElapsed= ( clock() - start ) / (CLOCKS_PER_SEC);

    printf("total time consumed: %lu millionsecond\n\n\n",timeElapsed);

 
}
