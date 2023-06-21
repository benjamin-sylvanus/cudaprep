
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
initPosition(int gid, double *dx2, int3 &Bounds, curandStatePhilox4_32_10_t *state,
                    double *SimulationParams, double4 *d4swc, int *nlut, int *NewIndex,
                    int3 &IndexSize, int size, int iter,int init_in, bool debug, double3 point);



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
diffusionTensor(double3 * A, double3 * xnot, double vsize, double *dx2, double * dx4, double3 * d2, int i, int gid, int iter, int size);

__device__ double3
/**
 *
 * @param nextpos next position
 * @param A current position
 * @param xi random unit vector (x,y,z,r)
 * @param step step size of simulation
 */
 setNextPos(double3 nextpos,double3 A,double4 xi, double step);


 /**
  * @brief Writes Results
  * @param w_swc
  * @param hostSimP
  * @param hostdx2
  * @param mdx2
  * @param hostdx4
  * @param mdx4
  * @param t
  * @param u_Reflections
  * @param u_uref
  * @param u_Sig0
  * @param u_SigRe
  * @param u_AllData
  * @param iter
  * @param size
  * @param nrow
  * @param sa_size
  * @param outpath
  * @return
  */
 __host__ void
 writeResults(double * w_swc, double * hostSimP, double * hostdx2, double * mdx2, double * hostdx4,
                                    double * mdx4, double * t, double * u_Reflections, double * u_uref, double * u_Sig0,
                                    double * u_SigRe, double * u_AllData,int iter, int size, int nrow, int sa_size,
                                    std::string outpath)

 /**
  * @brief Compute the next position of the particle
  * @param A current position
  * @param step step size
  * @param xi random unit vector
  * @param nextpos next position
  */
__device__ void computeNext(double3 &A, double &step, double4 &xi, double3 &nextpos, double &pi);

/**
 * @brief Checks if the particle is inside the connections listed for voxel
 * @param i_int3 size of the index array
 * @param test_lutvalue index of the voxel
 * @param nextpos position of the particle
 * @param NewIndex index array
 * @param d4swc swc array
 * @return true if particle is inside the connection
 */
__device__ bool checkConnections(int3 i_int3, int test_lutvalue, double3 nextpos, int *NewIndex, double4 *d4swc, double &fstep);

/**
 * @brief Checks the validity of the next position and updates the position and next position accordingly.
 * @param nextpos next position
 * @param pos current position
 * @param b_int3 bounding box
 * @param upper upper bound
 * @param lower lower bound
 * @param floorpos floor position
 * @param reflections reflection storage
 * @param uref reflection storage
 * @param gid particle id
 * @param i step id
 * @param size number of particles
 * @param iter number of iterations
 * @param flips reflection counter
 */
__device__ void validCoord(double3 &nextpos, double3 &pos, int3 &b_int3, int3 &upper, int3 &lower, int3 &floorpos,
                           double * reflections, double * uref, int gid, int i, int size, int iter, int * flips);


/**
 * @param u_dx2 diffusion tensor
 * @param u_dx4 diffusion tensor
 * @param u_SimP simulation parameters
 * @param u_D4Swc swc data
 * @param u_NewLut lookuptable
 * @param u_NewIndex index array
 * @param u_Flip reflection counter
 * @param simparam simulation parameters
 * @param swc_trim swc data
 * @param lut lookuptable
 * @param indexarr index array
 * @param bounds bounding box
 * @param nrow number of rows
 * @param prod product of dimensions
 * @param newindexsize size of index array
 * @param sa_size size of the saveall array
 * @param Nbvec number of vectors
 * @param timepoints number of timepoints
 * @param NC number of connections
 * @brief Sets up the data for the simulation
 */
__device__ __host__ void setup_data(double * u_dx2, double * u_dx4, double * u_SimP, double3 * u_D4Swc, int * u_NewLut,
                                    int * u_NewIndex, int * u_Flip, double * simparam, double3 * swc_trim, int * lut,
                                    int * indexarr, int * bounds, int nrow, int prod, int newindexsize, int sa_size, int Nbvec, int timepoints, int NC);
