#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<string.h>
#include"particle.h"
#include"find_neighbor.cpp"

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
  //  clock_t initializationtime = ( clock() - start ) / (CLOCKS_PER_SEC/1000);

   //     clock_t start1 = clock ();
  s.iterations(timeused);
  //    FILE* fp =fopen(s.fname,"w");
  //    FILE* fk;
  //      clock_t iterationstime= ( clock() - start1 ) / (CLOCKS_PER_SEC/1000);
        clock_t timeElapsed= ( clock() - start ) / (CLOCKS_PER_SEC);
/*    printf("\n\ninitialization time consumed: %lu millionsecond\n",initializationtime);
    printf("iteration time consumed: %lu millionsecond\n",iterationstime);
    printf("  sorting time consumed: %f millionsecond\n",timeused[2]);
    printf("  bforce calculation time consumed: %f millionsecond\n",timeused[0]);
    printf("  movement time consumed: %f millionsecond\n",timeused[1]);
 */
    printf("total time consumed: %lu millionsecond\n\n\n",timeElapsed);

 
}
