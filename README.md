# quicksort

sph:  with cpu neighbor searching

sphcuda: gpu neighbor searching

under sphcuda/ : cudasort.cu is written in cudaC and supposed to bin particles into cells with a spatial sequence, then perform 3D neighbor searching. 
syms.h defines the system parameter. particle.h defines the sph fluid class. 