# quicksort

sph:  with cpu neighbor searching

sphcuda: gpu neighbor searching (Makefile has been tested OS X 10.9.5, cuda 7.5)

under sphcuda/ : cudasort.cu is written in cudaC and supposed to bin particles into cells with a spatial sequence, then perform 3D neighbor searching. 
syms.h defines the system parameter. particle.h defines the sph fluid class. 

