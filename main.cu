#include <iostream>
#include "./src/simreader.h"
#include "./src/simulation.h"
#include "./src/particle.h"
#include "vector"
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda.h"
#include <cstring>
#include "curand.h"
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "device_launch_parameters.h"

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>
// #include <helper_cuda.h>

using std::cout;
using std::endl;

__global__ void setup_kernel(curandStatePhilox4_32_10_t *state, unsigned long seed) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void
doaflip(double *A, double *Bounds, int *LUT, int* IndexArray,double * SWC, bool *conditionalExample, curandStatePhilox4_32_10_t *state, int size, const int pairmax, int iter,
        bool debug) {

    int gid = threadIdx.x + blockDim.x * blockIdx.x;

    if (gid < size) {
        // if our id is within our range of values;
        // define variables for loop
        double4 xi;
        bool parstate[2];
        bool flag;
        double nextpos[3];
        int floorpos[3];
        bool upper[3];
        bool lower[3];
        bool completes;
        double permprob = 0.5;
        double step = 1;
        // init local state var
        curandStatePhilox4_32_10_t localstate = state[gid];
        // set particle initial position
        xi = curand_uniform4_double(&localstate);
        A[3 * gid + 0] = xi.x * Bounds[0];
        A[3 * gid + 1] = xi.y * Bounds[1];
        A[3 * gid + 2] = xi.z * Bounds[2];
        flag = 0;
        parstate[0]=1;
        parstate[1]=1;
        // iterate over steps
        for (int i = 0; i<iter; i++)
        {
          if (flag==0)
          {
            xi = curand_uniform4_double(&localstate);
            nextpos[0] = A[3 * gid + 0]  + ((2.0*xi.x-1.0)*step);
            nextpos[1] = A[3 * gid + 1]  + ((2.0*xi.y-1.0)*step);
            nextpos[2] = A[3 * gid + 2]  + ((2.0*xi.z-1.0)*step);

            // floor of next position -> check voxels
            floorpos[0] = (int) nextpos[0];
            floorpos[1] = (int) nextpos[1];
            floorpos[2] = (int) nextpos[2];

            // find voxels within range of lookup table
            upper[0] = floorpos[0] < Bounds[0];
            upper[1] = floorpos[1] < Bounds[1];
            upper[2] = floorpos[2] < Bounds[2];

            lower[0] = floorpos[0] >= 0;
            lower[1] = floorpos[1] >= 0;
            lower[2] = floorpos[2] >= 0;

            if (lower[0] && lower[1] && lower[2] && upper[0] && upper[1] && upper[2])
            {

              parstate[1] = 1;
            }
            else 
            {
              parstate[1] = 0;
            }

            // extract value of lookup @ index
            if (parstate[1])
            {
                int id = 0 + Bounds[0] * (Bounds[1] * floorpos[0] + floorpos[1]) + floorpos[2];
                int value = LUT[id];
                if (value>=0)
                {
                  // const int pmax_2 = pairmax+pairmax;
                  // extract value of indexarray @ index
                  int2 vindex;
                  for (int p = 0; p<pairmax; p++)
                  {
                    // for each pair...

                    vindex.x = IndexArray[pairmax*value + 2*p+0];
                    vindex.y = IndexArray[pairmax*value + 2*p+1];

                    // extract child parent values
                    double4 child; double4 parent;
                    child.x = (double) SWC[vindex.x*4 +0];
                    child.y = (double) SWC[vindex.x*4 +1];
                    child.z = (double) SWC[vindex.x*4 +2];
                    child.w = (double) SWC[vindex.x*4 +3];

                    parent.x =(double) SWC[vindex.y*4 +0];
                    parent.y =(double) SWC[vindex.y*4 +1];
                    parent.z =(double) SWC[vindex.y*4 +2];
                    parent.w =(double) SWC[vindex.y*4 +3];
                    // printf("In Range [particle: %1.1d] Voxel: [X=%1.1d,Y=%1.1d,Z=%1.1d] LUT@Index: [%2.1d] Index@LUT@Index: [%2.1d, %2.1d]\n",gid,floorpos[0],floorpos[1],floorpos[2],value,vindex.x,vindex.y);
                    // printf("[particle: %1.1d] child:[%.3f, %.3f, %.3f, %.3f] parent:[%.3f, %.3f, %.3f, %.3f]\n",gid,child.x,child.y,child.z,child.w,parent.x,parent.y,parent.z,parent.w);
                  
                    // check all connections

                    //distance squared between child parent
                    double dist2 = ((parent.x - child.x)*(parent.x - child.x)) + ((parent.y - child.y)*(parent.y - child.y)) + ((parent.z - child.z)*(parent.z - child.z));
                    
                    //call swc2v here?
                    //implement inline...
                    // printf("dist2: %.5f\n",dist2);
                  }
                  // printf("\n");
                }

            }
            // determine if step executes
            completes = xi.w<permprob;
            // printf("Completes [step,particle] (%d, %d): %d\n",i,gid,completes);
            if (completes && parstate[0] && parstate[1])
            {
              A[3 * gid + 0]=nextpos[0];
              A[3 * gid + 1]=nextpos[1];
              A[3 * gid + 2]=nextpos[2];
            }
            else 
            {
              flag = true;
            }
          }
          else
          {
            flag = false;
          }

          if (debug)
          {

          }
        }



      }
    }



void setupSimulation()
{
    std::string path = "/autofs/homes/009/bs244/cuda-workspace/hellocuda/cudaprep/data";
    simreader reader(&path);
    simulation sim(reader);
    sim.setStep_num(1000);
    sim.setParticle_num(10000);
    double *stepsize = sim.getStep_size();
}

void inithostlut(int * hostLookup, int prod, int bx, int by, int bz)
{
    memset(hostLookup, -1, prod * sizeof(int));
    int id0 = 0 + bx * (by * 0 + 1) + 1;
    int id1 = 0 + bx * (by * 1 + 1) + 1;
    int id2 = 0 + bx * (by * 2 + 1) + 1;
    int id3 = 0 + bx * (by * 3 + 1) + 1;
    int id4 = 0 + bx * (by * 4 + 1) + 1;
    int id01 = 0 + bx * (by * 0 + 2) + 1;
    int id11 = 0 + bx * (by * 1 + 2) + 1;
    int id21 = 0 + bx * (by * 2 + 2) + 1;
    int id31 = 0 + bx * (by * 3 + 2) + 1;
    int id41 = 0 + bx * (by * 4 + 2) + 1;
    hostLookup[id0] = 0;
    hostLookup[id1] = 1;
    hostLookup[id2] = 2;
    hostLookup[id3] = 3;
    hostLookup[id4] = 4;
    hostLookup[id01] = 5;
    hostLookup[id11] = 6;
    hostLookup[id21] = 7;
    hostLookup[id31] = 8;
    hostLookup[id41] = 9;
}

int main() {

    setupSimulation();
    system("clear");
    /**
     * Read Simulation and Initialize Object
     */

    std::string path = "/autofs/homes/009/bs244/cuda-workspace/hellocuda/cudaprep/data";
    simreader reader(&path);
    simulation sim(reader);

    std::vector<double> simulationparams = sim.getParameterdata();

    int size = 200000;
    int block_size = 128;
    int NO_BYTES = size * sizeof(double);
    dim3 block(block_size);
    dim3 grid((size/block.x)+1);
    int random_bytes = size * sizeof(double) * 3;

    double boundx = 5.0;
    double boundy = 5.0;
    double boundz = 5.0;

    int bx = (int) (boundx);
    int by = (int) (boundy);
    int bz = (int) (boundz);
    int prod = (int) (boundx * boundy *boundz);

    const int pairmax = 4;
    int npair = 10;

    /**
     * Host Section:
     * - Create Pointers
     * - Allocate Memory
     * - Set Values
     */
    // Create Host Pointers
    double *h_a;
    double *hostBounds;
    int *hostLookup;
    bool *hostLogicalVector;
    int * hostIndexArray;
    double * hostSWC;
    int indexsize = pairmax * npair;

    // Alloc Memory for Host Pointers
    h_a = (double *) malloc(random_bytes);
    hostBounds = (double *) malloc(3 * sizeof(double));
    hostLogicalVector = (bool *) malloc(size * sizeof(bool));
    hostLookup = (int *) malloc(prod * sizeof(int));
    hostIndexArray = (int *) malloc(pairmax*2*npair*sizeof(int));
    hostSWC = (double *) malloc(4*indexsize*sizeof(double));

    // Set Values for Host
    memset(hostIndexArray,0,pairmax*2*npair*sizeof(int));
    memset(hostSWC,0,indexsize * sizeof(double));
    
    for(int i = 0; i<npair; i++)
    {
      printf("HINDEX@I: ");
      for (int j=0; j<pairmax; j++)
      {
        hostIndexArray[pairmax*i+2*j+0]= pairmax*i + j+0;
        hostIndexArray[pairmax*i+2*j+1]= pairmax*i + j+1;
        printf("[%d,%d] ",hostIndexArray[pairmax*i+2*j+0],hostIndexArray[pairmax*i+2*j+1]);
      }
      printf("\n");
    }
    double counter = 0.8;
    for (int i = 0; i<indexsize; i++)
    {
      hostSWC[4*i+0] =  i * counter * counter;
      hostSWC[4*i+1] =  - i * counter * counter;
      hostSWC[4*i+2] =  ((double) i/2.0) * counter * counter;
      hostSWC[4*i+3] = ((double) i) + 3.0;
      if (counter >= 10.0)
      {
        counter= -10.2;
      }
      else 
      {
        counter+=2.900213466;
      }
    }

    memset(hostLogicalVector, false, size * sizeof(bool));
    hostBounds[0] = boundx; hostBounds[1] = boundy; hostBounds[2] = boundz;
    inithostlut(hostLookup, prod, bx ,by,bz);

    /**
     * Device Section:
     * - Create Pointers
     * - Allocate Memory
     * - Set Values
     */
    // Create Device Pointers
    curandStatePhilox4_32_10_t *deviceState;
    double *d_a;
    double *deviceBounds;
    int *deviceLookup;
    bool *deviceLogicalVector;
    int *deviceIndexArray;
    double *deviceSWC; 
    clock_t start = clock();
    // Allocate Memory on Device
    cudaMalloc((int **) &deviceLookup, prod * sizeof(int));
    cudaMalloc((int **) &deviceIndexArray, pairmax*2*npair*sizeof(int));
    cudaMalloc((double**) &deviceSWC, 4*indexsize*sizeof(double));
    cudaMalloc((double **) &d_a, random_bytes);
    cudaMalloc((double **) &deviceBounds, 3 * sizeof(double));
    cudaMalloc((bool **) &deviceLogicalVector, size * sizeof(bool));
    cudaMalloc((curandStatePhilox4_32_10_t **) & deviceState, size * sizeof(curandStatePhilox4_32_10_t));

    // Set Values for Device
    cudaMemcpy(deviceBounds, hostBounds, 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceLookup, hostLookup, prod * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceLogicalVector, hostLogicalVector, size * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceIndexArray, hostIndexArray, pairmax*2*npair*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(deviceSWC,hostSWC,4*indexsize*sizeof(double),cudaMemcpyHostToDevice);

    /**
     * Initalize Random Stream
     */
    setup_kernel<<<grid, block>>>(deviceState, 1);

    // option for printing in kernel
    bool debug = false;

    /**
     * Call Kernel
     */
    int iter = 1000000;



 


    doaflip<<<grid, block>>>(d_a, deviceBounds, deviceLookup, deviceIndexArray,deviceSWC, deviceLogicalVector, deviceState, size,pairmax,iter, debug);
    // Wait for results
    cudaDeviceSynchronize();

  

    /**
     * Copy Results From Device to Host
     */
    cudaMemcpy(h_a, d_a, random_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostLogicalVector, deviceLogicalVector, size * sizeof(bool), cudaMemcpyDeviceToHost);
    clock_t end = clock();
    double gpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("kernel took %f seconds\n", gpu_time_used);
    for (int i = 0; i < size; i++) {
        double x = h_a[3 * i + 0];
        double y = h_a[3 * i + 1];
        double z = h_a[3 * i + 2];
        // printf("x: %.4f \t y: %.4f \t z: %.4f\t Has Lookup Values: %d\n", x, y, z, hostLogicalVector[i]);
    }

    /**
     * Free Device Data
     */
    cudaFree(d_a);
    cudaFree(deviceBounds);
    cudaFree(deviceState);
    cudaFree(deviceLookup);
    cudaFree(deviceLogicalVector);
    /**
     * Free Host Data
     */
    free(h_a);
    free(hostBounds);
    free(hostLookup);
    free(hostLogicalVector);
    return 0;
}
