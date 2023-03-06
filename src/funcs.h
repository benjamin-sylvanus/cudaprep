
#include "vector"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cstring>
#include <curand.h>
#include <curand_kernel.h>
#include <fstream>

__device__ bool swc2v(double3 nextpos, double4 child, double4 parent, double dist);

__device__ int s2i(int3 i, int3 b);

__device__ double3 particleINITDEVICE2(int gid, double *dx2, int *Bounds, curandStatePhilox4_32_10_t *state,
                                       double *SimulationParams, double4 *d4swc, int *nlut, int *NewIndex,
                                       int *IndexSize, int size, int iter, bool debug);
