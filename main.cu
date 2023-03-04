#include "./src/simreader.h"
#include "./src/simulation.h"
#include "./src/particle.h"
#include "vector"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cstring>
#include <curand.h>
#include <curand_kernel.h>
#include "cuda_runtime_api.h"
#include <fstream>


using std::cout;
using std::cin;
using std::endl;

__device__ int s2i(int3 i, int3 b) {
    return 0 + b.x * (b.y * i.z + i.y) + i.x;
}

__device__ bool swc2v(double3 nextpos, double4 child, double4 parent, double dist) {
    double x0 = nextpos.x;
    double y0 = nextpos.y;
    double z0 = nextpos.z;

    double x1 = child.x;
    double y1 = child.y;
    double z1 = child.z;
    double r1 = child.w;
    double x2 = parent.x;
    double y2 = parent.y;
    double z2 = parent.z;
    double r2 = parent.w;

    double t = ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1) + (z0 - z1) * (z2 - z1)) / dist;
    double x = x1 + (x2 - x1) * t;
    double y = y1 + (y2 - y1) * t;
    double z = z1 + (z2 - z1) * t;

    bool list1;
    if (dist < (r1 * r1)) {
        list1 = false;
    } else {
        list1 = (x - x1) * (x - x2) + (y - y1) * (y - y2) + (z - z1) * (z - z2) < 0.0;
    }

    bool pos1;
    bool pos;
    if (list1) {
        double dist2 = ((x0 - x) * (x0 - x)) + ((y0 - y) * (y0 - y)) + ((z0 - z) * (z0 - z));

        /**
         * @brief calculation for tangent line
         * <li> r = r1 + sqrt((x-x1).^2 + (y-y1).^2 + (z-z1).^2) / sqrt((x2-x1)^2+(y2-y1)^2+(z2-z1)^2) * (r2-r1) </li>
         * <li> r = ( c + r2 ) / (sqrt ( 1 - ( |r1-r2 | / l ) ) </li>
         * <li>c = ( |r1 - r2| * l ) / L </li>
        *
        */
        double rd = abs(r1 - r2);

        // distance from orthogonal vector to p2
        double l = sqrt(((x - x2) * (x - x2)) + ((y - y2) * (y - y2)) + ((z - z2) * (z - z2)));

        // distance from p1 -> p2
        double L = sqrt(dist);

        double c = (rd * l) / L;
        double r = (c + r2) / sqrt(1 - ((rd / L) * (rd / L)));
        pos1 = dist2 < (r * r);
        pos = pos1;
    } else {
        pos1 = ((((x0 - x1) * (x0 - x1)) + ((y0 - y1) * (y0 - y1)) + ((z0 - z1) * (z0 - z1))) < (r1 * r1)) ||
               ((((x0 - x2) * (x0 - x2)) + ((y0 - y2) * (y0 - y2)) + ((z0 - z2) * (z0 - z2))) < ((r2 * r2)));
        pos = pos1;
    }

    return pos;
}

// need pairs to evaluate, pointers to LUT, SWC, index, GID?
/**
 * @brief
 * if it evaluates to inside we write, doesnt matter if values are overwritten.
 * @param size
 * @return __global__
 */

__global__ void parallelpairs(int2 * parstate,int3 i_int3,double3 nextpos, double4 * d4swc, int * NewIndex, int test_lutvalue, int particleIndex)
{
    // read
    int gid = threadIdx.x + blockDim.x * blockIdx.x;

    if (gid<i_int3.z)
    {

        // printf("GID: %d\n",gid);
        //probably need to init arrays for each pair
        double4 child;
        double4 parent;
        int2 vindex;
        int3 c_new = make_int3(test_lutvalue, 0, gid);
        int3 p_new = make_int3(test_lutvalue, 1, gid);
        vindex.x = NewIndex[s2i(c_new, i_int3)] - 1;
        vindex.y = NewIndex[s2i(p_new, i_int3)] - 1;
        if ((vindex.x) != -1)
        {
            child.x = (double) d4swc[vindex.x].x;
            child.y = (double) d4swc[vindex.x].y;
            child.z = (double) d4swc[vindex.x].z;
            child.w = (double) d4swc[vindex.x].w;
            // printf("CHILD \tx:%2.4f y:%2.4f z:%2.4f r:%2.4f\n", child.x, child.y, child.z, child.w);

            // extract parent values
            parent.x = (double) d4swc[vindex.y].x;
            parent.y = (double) d4swc[vindex.y].y;
            parent.z = (double) d4swc[vindex.y].z;
            parent.w = (double) d4swc[vindex.y].w;
            // printf("PARENT \tx:%2.4f y:%2.4f z:%2.4f r:%2.4f\n", parent.x, parent.y, parent.z, parent.w);
            //distance squared between child parent
            double dist2 = ((parent.x - child.x) * (parent.x - child.x)) + ((parent.y - child.y) * (parent.y - child.y)) + ((parent.z - child.z) * (parent.z - child.z));

            // determine whether particle is inside this connection
            bool inside = swc2v(nextpos, child, parent, dist2);

            // if it is inside the connection we don't need to check the remaining.
            if (inside) {
                // update the particles state
                parstate[particleIndex].y = 1;
                // main.cu(293) : Error: a pointer to local memory cannot be passed to a launch as an argument
            }
            __syncthreads();

        }

    }
}


__global__ void setup_kernel(curandStatePhilox4_32_10_t *state, unsigned long seed) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}


__device__ void particleINITDEVICE(int gid,double *A, double *dx2, int *Bounds, curandStatePhilox4_32_10_t *state, double *SimulationParams, double4 *d4swc, int2 * parstate, int *nlut, int *NewIndex, int *IndexSize, int size, const int pairmax, int iter, bool debug)
{
      int3 gx;
      gx.x = 3 * gid + 0;
      gx.y = 3 * gid + 1;
      gx.z = 3 * gid + 2;

      double3 nextpos;

      int3 upper;
      int3 lower;
      int3 floorpos;
      int3 b_int3;
      b_int3.x = Bounds[0];
      b_int3.y = Bounds[1];
      b_int3.z = Bounds[2];
      int3 i_int3;
      i_int3.x = IndexSize[0];
      i_int3.y = IndexSize[1];
      i_int3.z = IndexSize[2];
      curandStatePhilox4_32_10_t localstate = state[gid];
      double4 xrandom;
      parstate[gid].y=0;
      int ntrys=0;
      int nloops=0;
      bool cont = true;
      while (cont)
      {
        // init local state var

        xrandom = curand_uniform4_double(&localstate);

        // set particle initial position
        A[gx.x] = xrandom.x * (double) (Bounds[0]-4);
        A[gx.y] = xrandom.y * (double) (Bounds[1]-4);
        A[gx.z] = xrandom.z * (double) (Bounds[2]-4);

        // floor of position -> check voxels
        floorpos.x = (int) A[gx.x]; floorpos.y = (int) A[gx.y]; floorpos.z = (int) A[gx.z];

        // upper bounds of lookup table
        upper.x = floorpos.x < (Bounds[0]-4);
        upper.y = floorpos.y < (Bounds[1]-4);
        upper.z = floorpos.z < (Bounds[2]-4);

        // lower bounds of lookup table
        lower.x = floorpos.x >= 4;
        lower.y = floorpos.y >= 4;
        lower.z = floorpos.z >= 4;

        // position inside the bounds of volume -> state of next position true : false
        parstate[gid].x = (lower.x && lower.y && lower.z && upper.x && upper.y && upper.z) ? 1 : 0;

    if (parstate[gid].x)
    {
          // scuffed sub2ind function
          int id_test = s2i(floorpos, b_int3);
          int test_lutvalue = nlut[id_test];
          nextpos = make_double3(A[gx.x],A[gx.y],A[gx.z]);
          parallelpairs<<<2,1>>>(parstate, i_int3, nextpos , d4swc, NewIndex, test_lutvalue, gid);
          // __syncthreads();
          if (parstate[gid].y == 1)
          {
            cont = false;
          }
          else
          {
            ntrys++;
          }
    }
    if(parstate[gid].y == 1)
    {
        printf("Particle %d took %d trys\n",gid,ntrys);
        cont = false;
    }
  }
}


__global__ void particleInit(double *A, double *dx2, int *Bounds, curandStatePhilox4_32_10_t *state, double *SimulationParams, double4 *d4swc, int2 * parstate, int *nlut, int *NewIndex, int *IndexSize, int size, const int pairmax, int iter, bool debug)
{
    int gid = threadIdx.x + blockDim.x * blockIdx.x;

    if (gid < size)
    {
        int3 gx;
        gx.x = 3 * gid + 0;
        gx.y = 3 * gid + 1;
        gx.z = 3 * gid + 2;

        double3 nextpos;

        int3 upper;
        int3 lower;
        int3 floorpos;
        int3 b_int3;
        b_int3.x = Bounds[0];
        b_int3.y = Bounds[1];
        b_int3.z = Bounds[2];
        int3 i_int3;
        i_int3.x = IndexSize[0];
        i_int3.y = IndexSize[1];
        i_int3.z = IndexSize[2];
        double4 xi;

        // init local state var
        curandStatePhilox4_32_10_t localstate = state[gid];
        xi = curand_uniform4_double(&localstate);
        double4 xrandom;
        parstate[gid].y=0;
        int ntrys=0;
        int nloops=0;
        bool cont = true;
        while (cont)
        {
          xrandom = curand_uniform4_double(&localstate);

          // set particle initial position
          A[gx.x] = xrandom.x * (double) Bounds[0];
          A[gx.y] = xrandom.y * (double) Bounds[1];
          A[gx.z] = xrandom.z * (double) Bounds[2];

          // floor of position -> check voxels
          floorpos.x = (int) A[gx.x]; floorpos.y = (int) A[gx.y]; floorpos.z = (int) A[gx.z];

          // upper bounds of lookup table
          upper.x = floorpos.x < Bounds[0];
          upper.y = floorpos.y < Bounds[1];
          upper.z = floorpos.z < Bounds[2];

          // lower bounds of lookup table
          lower.x = floorpos.x >= 0;
          lower.y = floorpos.y >= 0;
          lower.z = floorpos.z >= 0;

          // position inside the bounds of volume -> state of next position true : false
          parstate[gid].x = (lower.x && lower.y && lower.z && upper.x && upper.y && upper.z) ? 1 : 0;

          if (parstate[gid].x)
          {
              // scuffed sub2ind function
              int id_test = s2i(floorpos, b_int3);
              int test_lutvalue = nlut[id_test];
              nextpos = make_double3(A[gx.x],A[gx.y],A[gx.z]);
              parallelpairs<<<2,2>>>(parstate, i_int3, nextpos , d4swc, NewIndex, test_lutvalue, gid);
              // __syncthreads();
              if (parstate[gid].y == 1)
              {
                cont = false;
              }
              else
              {
                ntrys++;
              }
            }

      if(parstate[gid].y == 1)
      {
        printf("Particle %d took %d trys\n",gid,ntrys);
        cont = false;
      }

        }
    }
}


__global__ void
simulate(double *A, double *dx2, int *Bounds, curandStatePhilox4_32_10_t *state, double *SimulationParams, double4 *d4swc, int2 * parstate, int *nlut, int *NewIndex, int *IndexSize, int size, const int pairmax, int iter, bool debug)
{
    int gid = threadIdx.x + blockDim.x * blockIdx.x;

    if (gid < size) {
        double particle_num = SimulationParams[0];
        double step_num = SimulationParams[1];
        double step_size = SimulationParams[2];
        double perm_prob = SimulationParams[3];
        double init_in = SimulationParams[4];
        double D0 = SimulationParams[5];
        double d = SimulationParams[6];
        double scale = SimulationParams[7];
        double tstep = SimulationParams[8];
        double vsize = SimulationParams[9];

        int3 gx;
        gx.x = 3 * gid + 0;
        gx.y = 3 * gid + 1;
        gx.z = 3 * gid + 2;

        // if our id is within our range of values;
        // define variables for loop
        double4 xi;
        double3 nextpos;
        double3 xnot;
        double3 d2;

        int3 upper;
        int3 lower;
        int3 floorpos;
        int3 b_int3;
        b_int3.x = Bounds[0];
        b_int3.y = Bounds[1];
        b_int3.z = Bounds[2];

        int3 i_int3;
        i_int3.x = IndexSize[0];
        i_int3.y = IndexSize[1];
        i_int3.z = IndexSize[2];

        bool completes;
        bool flag;

        double permprob = 0.0;
        double step = step_size;

        // init local state var
        curandStatePhilox4_32_10_t localstate = state[gid];
        xi = curand_uniform4_double(&localstate);

        /**
         * @brief Initialize Random Positions:
         *
        */

        particleINITDEVICE(gid,A,dx2,Bounds,state,SimulationParams, d4swc, parstate, nlut, NewIndex, IndexSize, size,pairmax, iter, debug);


        // record initial position
        xnot.x = A[gx.x];
        xnot.y = A[gx.y];
        xnot.z = A[gx.z];
        // flag is initially false
        flag = 0;

        // state is based on intialization conditions if particles are required to start inside then parstate -> [1,1]
        parstate[gid].x = 1;
        parstate[gid].y = 1;

        // iterate over steps
        for (int i = 0; i < iter; i++) {


            if (flag == 0) {
                xi = curand_uniform4_double(&localstate);
                nextpos.x = A[gx.x] + ((2.0 * xi.x - 1.0) * step);
                nextpos.y = A[gx.y] + ((2.0 * xi.y - 1.0) * step);
                nextpos.z = A[gx.z] + ((2.0 * xi.z - 1.0) * step);

                // floor of next position -> check voxels
                floorpos.x = (int) nextpos.x;
                floorpos.y = (int) nextpos.y;
                floorpos.z = (int) nextpos.z;

                // find voxels within range of lookup table

                // upper bounds of lookup table
                upper.x = floorpos.x < Bounds[0];
                upper.y = floorpos.y < Bounds[1];
                upper.z = floorpos.z < Bounds[2];

                // lower bounds of lookup table
                lower.x = floorpos.x >= 0;
                lower.y = floorpos.y >= 0;
                lower.z = floorpos.z >= 0;

                // position inside the bounds of volume -> state of next position true : false
                parstate[gid].y = (lower.x && lower.y && lower.z && upper.x && upper.y && upper.z) ? 1 : 0;
                // extract value of lookup @ index
                if (parstate[gid].y) {
                    // reset particle state for next conditionals
                    parstate[gid].y = 0;

                    // scuffed sub2ind function
                    int id_test = s2i(floorpos, b_int3);
                    int test_lutvalue = nlut[id_test];

                    /**
                     * @brief Indexing Index Array
                     * <li> N x 2 x M </li>
                     * <li> Index Array subscripted indices can be passed to s2i. </li>
                     * <li> Our lookuptable returns x value of index array </li>
                     * <li> Each X is a 2xM array; </li>
                     * <li> We take child parent pairs from 0:M using for loop; </li>
                     * <li> index formula: </li>
                     * \b stride + dx * (dy * z + y) + x;
                     */


                    //kernel call
                    // int2 parstate,double3 nextpos, double4 * d4swc, int * NewIndex, int test_lutvalue, int size
                    parallelpairs<<<2,2>>>(parstate, i_int3, nextpos , d4swc, NewIndex, test_lutvalue, gid);
                    __syncthreads();
                }

                // determine if step executes
                completes = xi.w < permprob;
                // printf("Completes [step,particle] (%d, %d): %d\n",i,gid,completes);

                if (parstate[gid].y)
                {
                    // printf("check\n");
                    A[gx.x] = nextpos.x;
                    A[gx.y] = nextpos.y;
                    A[gx.z] = nextpos.z;
                }
                else
                {
                    if (completes && parstate[gid].x) {
                        printf("ESCAPEE ALERT!\n");
                        A[gx.x] = nextpos.x;
                        A[gx.y] = nextpos.y;
                        A[gx.z] = nextpos.z;
                    }
                    if (!completes && parstate[gid].x && !parstate[gid].y) {
                        // printf("REJECTION ALERT!\n");
                        flag = true;
                    }
                }
            }
            else
            {
                flag = false;
            }

            // d2.x = (A[gx.x] - xnot.x) * vsize;
            // d2.y = (A[gx.y] - xnot.y) * vsize;
            // d2.z = (A[gx.z] - xnot.z) * vsize;

            d2.x = abs((A[gx.x] - xnot.x) * vsize);
            d2.y = abs((A[gx.y] - xnot.y) * vsize);
            d2.z = abs((A[gx.z] - xnot.z) * vsize);

            // diffusion tensor
            atomicAdd(&dx2[6 * i + 0], d2.x * d2.x);
            atomicAdd(&dx2[6 * i + 1], d2.x * d2.y);
            atomicAdd(&dx2[6 * i + 2], d2.x * d2.z);
            atomicAdd(&dx2[6 * i + 3], d2.y * d2.y);
            atomicAdd(&dx2[6 * i + 4], d2.y * d2.z);
            atomicAdd(&dx2[6 * i + 5], d2.z * d2.z);

        }

    }
  }

void setupSimulation() {
    std::string path = "/autofs/space/symphony_002/users/BenSylvanus/cuda/cudaprep/data";
    simreader reader(&path);
    simulation sim(reader);
    sim.setStep_num(100);
    sim.setParticle_num(100);
    double *stepsize = sim.getStep_size();
}


int main() {

    setupSimulation();
    system("clear");

    /**
     * Read Simulation and Initialize Object
     */
    std::string path = "/homes/9/bs244/Desktop/cudacodes/temp/cudaprep/data";
    simreader reader(&path);
    simulation sim(reader);
    double simparam[10];
    std::vector<double> simulationparams = sim.getParameterdata();

    for (int i = 0; i < 10; i++) {
        double value = simulationparams[i];
        simparam[i] = value;
        printf("%f\t", simulationparams[i]);
    }
    printf("\n");

    /**
    todo figure out why lower values of size and iteration causes bug.
    -- This appears to be the cutoff
    int size = 65;
    int iter = 65;
    */
    int size = 1000;
    int iter = 1000;
    cout << "Enter Number of Particles: \n";
    cin >> size;
    cout << "Enter Number of Steps" << '\n';
    cin >> iter;
    // int size = 10;
    // int iter = 10000;
    int block_size = 128;
    dim3 block(block_size);
    dim3 grid((size / block.x) + 1);

    // we could convert our devices random vector to a double3
    int random_bytes = 3 * size * sizeof(double);

    // todo: figure out why different bound size gives bug
    std::vector <uint64_t> bounds = sim.getbounds();
    int boundx = (int) bounds[0];
    int boundy = (int) bounds[1];
    int boundz = (int) bounds[2];

    int prod = (int) (boundx * boundy * boundz);

    std::vector<double> r_swc = sim.getSwc();
    int nrow = r_swc.size() / 6;
    printf("NRows: %d\n", r_swc.size() / 6);

    /**
     * @brief Indexing SWC ARRAY
     * index + (dx*0:5);
     * row+(nrow*col)
     * Example: nrow = 10; get all elements from row 1;
     * swc[1,0:5] ---->
     * <li> swc(1,0) = swc[1+10*0];</li>
     * <li> swc(1,1) = swc[1+10*1];</li>
     * <li> swc(1,2) = swc[1+10*2];</li>
     * <li> swc(1,3) = swc[1+10*3];</li>
     * <li> swc(1,4) = swc[1+10*4];</li>
     * <li> swc(1,5) = swc[1+10*5];</li>
     */
    int doffset = 2;
    printf("[0,:]= %f \t %f \t %f \t %f \t %f \t %f\n", r_swc[0 + doffset * 0], r_swc[0 + doffset * 1],
           r_swc[0 + doffset * 2], r_swc[0 + doffset * 3], r_swc[0 + doffset * 4], r_swc[0 + doffset * 5]);
    printf("[1,:]= %f \t %f \t %f \t %f \t %f \t %f\n", r_swc[1 + doffset * 0], r_swc[1 + doffset * 1],
           r_swc[1 + doffset * 2], r_swc[1 + doffset * 3], r_swc[1 + doffset * 4], r_swc[1 + doffset * 5]);
    // 0: [1 6 6 6 2 1] 1: [2 16 6 6 2 1]

    //we only need the x y z r of our swc array.
    double4 swc_trim[nrow];
    for (int i = 0; i < nrow; i++) {
        swc_trim[i].x = r_swc[i + nrow * 1];
        swc_trim[i].y = r_swc[i + nrow * 2];
        swc_trim[i].z = r_swc[i + nrow * 3];
        swc_trim[i].w = r_swc[i + nrow * 4];
    }
    printf("Trimmed SWC[0,:] = %f \t %f \t %f \t %f \n", swc_trim[0].x, swc_trim[0].y, swc_trim[0].z, swc_trim[0].w);
    printf("Trimmed SWC[0,:] = %f \t %f \t %f \t %f \n", swc_trim[1].x, swc_trim[1].y, swc_trim[1].z, swc_trim[1].w);
    const int pairmax = 4;
    int npair = 10;

    std::vector <uint64_t> lut = sim.getLut();

    // stride + bx * (by * y + z) + x
    int id0 = 0 + (boundx) * ((boundy) * 2 + 2) + 3;
    printf("lut[%d]: %d\n", id0, lut[id0]);
    std::vector <uint64_t> indexarr = sim.getIndex();

    /**
     * @brief Lookup Table Summary
     * linearindex = stride + bx * (by * z + y) + x
     * voxel coord: (x,y,z);
    */
    std::vector <std::vector<uint64_t>> arrdims = sim.getArraydims();
    std::vector <uint64_t> swc_dims = arrdims[0];
    std::vector <uint64_t> lut_dims = arrdims[1];
    std::vector <uint64_t> index_dims = arrdims[2];
    std::vector <uint64_t> pairs_dims = arrdims[3];
    std::vector <uint64_t> bounds_dims = arrdims[4];
    printf("%d \t %d \t %d\n", index_dims[0], index_dims[1], index_dims[2]);
    int newindexsize = index_dims[0] * index_dims[1] * index_dims[2];

    /**
     * Host Section:
     * - Create Pointers
     * - Allocate Memory
     * - Set Values
     */
    // Create Host Pointers
    double *h_a;
    int *hostBounds;
    double *hostdx2;
    double *hostSimP;
    int *hostNewLut;
    int *hostNewIndex;
    int *hostIndexSize;
    double4 *hostRandomParticle;

    int2 *hostParticleState;
    double4 *hostD4Swc;

    // Alloc Memory for Host Pointers
    h_a = (double *) malloc(random_bytes);
    hostBounds = (int *) malloc(3 * sizeof(int));
    hostdx2 = (double *) malloc(6 * iter * sizeof(double));
    hostSimP = (double *) malloc(10 * sizeof(double));
    hostD4Swc = (double4 *) malloc(nrow * sizeof(double4));
    hostNewLut = (int *) malloc(prod * sizeof(int));
    hostNewIndex = (int *) malloc(newindexsize * sizeof(int));
    hostIndexSize = (int *) malloc(3 * sizeof(int));
    hostParticleState=(int2 *) malloc(size*sizeof(int2));
    hostRandomParticle = (double4 *) malloc(size * 10 * sizeof(double4));


    // Set Values for Host
    memset(hostdx2, 0.0, 6 * iter * sizeof(double));
    memset(hostParticleState,0,size*sizeof(int2));
    memset(hostRandomParticle,0.0,size*10*sizeof(double4));


    for (int i = 0; i < 3; i++) {
        int value = index_dims[i];
        hostIndexSize[i] = value;
    }

    for (int i = 0; i < indexarr.size(); i++) {
        int value = indexarr[i];
        hostNewIndex[i] = value;
    }

    for (int i = 0; i < prod; i++) {
        int value = lut[i];
        hostNewLut[i] = value;
    }

    for (int i = 0; i < nrow; i++) {
        hostD4Swc[i].x = swc_trim[i].x;
        hostD4Swc[i].y = swc_trim[i].y;
        hostD4Swc[i].z = swc_trim[i].z;
        hostD4Swc[i].w = swc_trim[i].w;
    }


    for (int i = 0; i < 10; i++) {
        hostSimP[i] = simparam[i];
    }

    hostBounds[0] = boundx;
    hostBounds[1] = boundy;
    hostBounds[2] = boundz;


    /**
     * Device Section:
     * - Create Pointers
     * - Allocate Memory
     * - Set Values
     */
    // Create Device Pointers
    curandStatePhilox4_32_10_t *deviceState;
    double *d_a;
    int *deviceBounds;

    double *devicedx2;
    double *deviceSimP;
    int2 *deviceParticleState;
    double4 *deviced4Swc;
    // double4 *deviceRandomParticle;

    int *deviceNewLut;
    int *deviceNewIndex;
    int *deviceIndexSize;

    clock_t start = clock();
    // Allocate Memory on Device
    cudaMalloc((double **) &d_a, random_bytes);
    cudaMalloc((int **) &deviceBounds, 3 * sizeof(int));
    cudaMalloc((curandStatePhilox4_32_10_t * *) & deviceState, size * sizeof(curandStatePhilox4_32_10_t));
    cudaMalloc((double **) &devicedx2, 6 * iter * sizeof(double));
    cudaMalloc((double **) &deviceSimP, 10 * sizeof(double));
    cudaMalloc((double4 * *) & deviced4Swc, nrow * sizeof(double4));
    cudaMalloc((int **) &deviceNewLut, prod * sizeof(int));
    cudaMalloc((int **) &deviceNewIndex, newindexsize * sizeof(int));
    cudaMalloc((int **) &deviceIndexSize, 3 * sizeof(int));
    cudaMalloc((int2 * *) &deviceParticleState,size*sizeof(int2));
    // cudaMalloc((double4 * *) &deviceRandomParticle,size*10*sizeof(double4));

    // Set Values for Device
    cudaMemcpy(deviceBounds, hostBounds, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devicedx2, hostdx2, 6 * iter * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceSimP, hostSimP, 10 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviced4Swc, hostD4Swc, nrow * sizeof(double4), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceNewLut, hostNewLut, prod * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceNewIndex, hostNewIndex, newindexsize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceIndexSize, hostIndexSize, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceParticleState, hostParticleState, size * sizeof(int2), cudaMemcpyHostToDevice);
    // cudaMemcpy(deviceRandomParticle, hostRandomParticle, size * 10 * sizeof(double4), cudaMemcpyHostToDevice);

    /**
     * Initalize Random Stream
     */
    setup_kernel<<<grid, block>>>(deviceState, 1);

    // option for printing in kernel
    bool debug = false;

    /**
     * Call Kernel
    */

    // particleInit(double *A, double *dx2, int *Bounds, curandStatePhilox4_32_10_t *state, double *SimulationParams, double4 *d4swc, int2 * parstate, int *nlut, int *NewIndex, int *IndexSize, int size, const int pairmax, int iter, bool debug);
    // clock_t start_particleInit = clock();
    // particleInit<<<grid,block>>>(d_a, devicedx2, deviceBounds, deviceState, deviceSimP, deviced4Swc, deviceParticleState, deviceNewLut, deviceNewIndex,
    //                           deviceIndexSize, size, pairmax, iter, debug);
    //
    // cudaDeviceSynchronize();
    // clock_t stop_particleInit = clock();
    // double particle_time_used = ((double) (stop_particleInit - start_particleInit)) / CLOCKS_PER_SEC;
    // printf("Particle Init took %f seconds\n", particle_time_used);


    simulate<<<grid, block>>>(d_a, devicedx2, deviceBounds, deviceState, deviceSimP, deviced4Swc, deviceParticleState, deviceNewLut, deviceNewIndex,
                              deviceIndexSize, size, pairmax, iter, debug);
    // Wait for results
    cudaDeviceSynchronize();
    clock_t end = clock();
    double gpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("kernel took %f seconds\n", gpu_time_used);

    /**
     * Copy Results From Device to Host
     */
    cudaMemcpy(h_a, d_a, random_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostdx2, devicedx2, 6 * iter * sizeof(double), cudaMemcpyDeviceToHost);
    end = clock();
    gpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("kernel took %f seconds\n", gpu_time_used);

    for(int i=0; i<size; i++)
    {
        // printf("x: %f \t y: %f \t z: %f\n",h_a[3*i+0], h_a[3*i+1], h_a[3*i+2]);

    }

    for (int i = 0; i < iter; i+=100)
    {

       // printf("xx: %f \t xy: %f \t xz: %f\t yy: %f\t yz: %f\t zz: %f\t \n", hostdx2[i * 6 + 0],
          // hostdx2[i * 6 + 1], hostdx2[i * 6 + 2], hostdx2[i * 6 + 3], hostdx2[i * 6 + 4], hostdx2[i * 6 + 5]);
    }



    /**
     * Free Device Data
     */

    cudaFree(d_a);
    cudaFree(deviceBounds);
    cudaFree(deviceState);
    cudaFree(devicedx2);
    cudaFree(deviceSimP);
    cudaFree(deviced4Swc);
    cudaFree(deviceNewIndex);
    cudaFree(deviceIndexSize);
    cudaFree(deviceParticleState);


    /**
    * Write Results to file
    We need to write
    hostdx2

    */


    // Save results

    double t[iter];
    double tstep = hostSimP[8];
    double mdx_2[6*iter];
    for (int  i = 0; i<iter; i++)
    {
      t[i] = tstep*i;
      mdx_2[6*i+0] = (hostdx2[6*i+0]/size)/(2.0*t[i]);
      mdx_2[6*i+1] = (hostdx2[6*i+1]/size)/(2.0*t[i]);
      mdx_2[6*i+2] = (hostdx2[6*i+2]/size)/(2.0*t[i]);
      mdx_2[6*i+3] = (hostdx2[6*i+3]/size)/(2.0*t[i]);
      mdx_2[6*i+4] = (hostdx2[6*i+4]/size)/(2.0*t[i]);
      mdx_2[6*i+5] = (hostdx2[6*i+5]/size)/(2.0*t[i]);
      printf("%f\n",t[i]);
    }



    std::ofstream fdx2out("/autofs/space/symphony_002/users/BenSylvanus/cuda/cudaprep/results/dx2.txt");
    std::ofstream fmdx2out("/autofs/space/symphony_002/users/BenSylvanus/cuda/cudaprep/results/mdx2.txt");

    fdx2out.precision(15);
    fmdx2out.precision(15);

    for (int i=0; i<iter; i++) {

        for (int j=0;j<6;j++) {

            if (j==5) {
                fdx2out<<hostdx2[i*6+j]<<endl;
            }
            else {
                fdx2out<<hostdx2[i*6+j]<<"\t";
            }
            if (j==5) {
                fmdx2out<<mdx_2[i*6+j]<<endl;
            }
            else {
                fmdx2out<<mdx_2[i*6+j]<<"\t";
            }
        }
    }

    fdx2out.close();
    fmdx2out.close();




    /**
     * Free Host Data
     */
    free(h_a);
    free(hostBounds);
    free(hostdx2);
    free(hostSimP);
    free(hostD4Swc);
    free(hostNewIndex);
    free(hostIndexSize);
    free(hostParticleState);
    return 0;
}
