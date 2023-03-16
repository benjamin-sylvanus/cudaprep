
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

/**
 * Determine if particle is inside connection
 * @param nextpos particle position
 * @param child child node of connection
 * @param parent parent node of connection
 * @param dist distance from child to parent
 * @return inside? 1:0
 */
__device__ bool swc2v(double3 nextpos, double4 child, double4 parent, double dist);

/**
 * Subscript to Linear Index
 * @param i subscripted index: (row col page) -> (i.x i.y i.z)
 * @param b size of array: (nrows ncols npages) -> (b.x b.y b.z)
 * @return linear index
 */
__device__ __host__ int s2i(int3 i, int3 b);

__device__ double3
/**
 * Generates an initial random position inside cell
 * @param gid repr particle id: gid = threadIdx.x + blockDim.x * blockIdx.x;
 * @param dx2 diffusion tensor
 * @param Bounds bounds of our geometry
 * @param state curand handle
 * @param SimulationParams double array of simulation parameters
 * @param d4swc double4 array of swc data
 * @param nlut lookuptable of simulation
 * @param NewIndex Index Array of simulation
 * @param IndexSize Dimensions of Index Array
 * @param size alias of npar: Number of Particles
 * @param iter alias of nstep: Number of Steps
 * @param debug display calculations?
 * @return initial position
 */
initPosition(int gid, double *dx2, int *Bounds, curandStatePhilox4_32_10_t *state,
                    double *SimulationParams, double4 *d4swc, int *nlut, int *NewIndex,
                    int *IndexSize, int size, int iter,int init_in, bool debug);
                    


__device__ void
/**
 * Calculate the Diffusion Tensor
 * @param A current particle position
 * @param xnot initial particle position
 * @param vsize voxel size
 * @param dx2 Diffusion Tensor
 * @param savedata pointer to saved data.
 * @param d2 particle displacement (scaled)
 * @param i step
 */
diffusionTensor(double3 A, double3 xnot, double vsize, double *dx2, double *savedata, double3 d2, int i, int gid, int iter, int size);

__device__ double3
/**
 *
 * @param nextpos next position
 * @param A current position
 * @param xi random unit vector (x,y,z,r)
 * @param step step size of simulation
 */
 setNextPos(double3 nextpos,double3 A,double4 xi, double step);
