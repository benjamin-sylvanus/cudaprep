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

// ********** cuda kernel **********
__device__ double atomAdd(double *address, double val) {
    unsigned long long int *address_as_ull =
            (unsigned long long int *) address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
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

__global__ void setup_kernel(curandStatePhilox4_32_10_t *state, unsigned long seed) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void
simulate(double *A, double *dx2, double *Bounds, int *LUT, int *IndexArray, double *SWC, bool *conditionalExample,
         curandStatePhilox4_32_10_t *state, int size, const int pairmax, int iter,
         bool debug) {

    int gid = threadIdx.x + blockDim.x * blockIdx.x;

    if (gid < size) {
        int3 gx;
        gx.x = 3 * gid + 0;
        gx.y = 3 * gid + 1;
        gx.z = 3 * gid + 2;

        // if our id is within our range of values;
        // define variables for loop
        double4 xi;
        int2 vindex;
        double4 child;
        double4 parent;

        double3 nextpos;
        double3 xnot;

        int3 upper;
        int3 lower;
        int3 floorpos;

        int2 parstate;
        bool completes;
        bool flag;

        double permprob = 0.0;
        double step = 1;
        double res = 0.5;
        double dx;
        double dy;
        double dz;
        double dist2;

        // init local state var
        curandStatePhilox4_32_10_t localstate = state[gid];

        // set particle initial position
        xi = curand_uniform4_double(&localstate);

        // todo: random initial state
        A[gx.x] = xi.x * Bounds[0];
        A[gx.y] = xi.y * Bounds[1];
        A[gx.z] = xi.z * Bounds[2];

        // record initial position
        xnot.x = A[gx.x];
        xnot.y = A[gx.y];
        xnot.z = A[gx.z];

        // flag is initially false
        flag = 0;

        // state is based on intialization conditions if particles are required to start inside then parstate -> [1,1]
        parstate.x = 1;
        parstate.y = 1;

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
                parstate.y = (lower.x && lower.y && lower.z && upper.x && upper.y && upper.z) ? 1 : 0;

                // extract value of lookup @ index
                if (parstate.y) {
                    // reset particle state for next conditionals
                    parstate.y = 0;

                    // scuffed sub2ind function
                    // todo: check this implementation
                    int id = 0 + Bounds[0] * (Bounds[1] * floorpos.x + floorpos.y) + floorpos.z;

                    // warn this is unchecked
                    int value = LUT[id];
                    if (value >= 0) {
                        // const int pmax_2 = pairmax+pairmax;
                        // extract value of indexarray @ index

                        for (int p = 0; p < pairmax; p++) {
                            // extract the childId and parentId
                            vindex.x = IndexArray[pairmax * value + 2 * p + 0];
                            vindex.y = IndexArray[pairmax * value + 2 * p + 1];
                            // check validity of connection given some index array values will be empty
                            if ((vindex.x) != -1) {
                                // extract child values
                                child.x = (double) SWC[vindex.x * 4 + 0];
                                child.y = (double) SWC[vindex.x * 4 + 1];
                                child.z = (double) SWC[vindex.x * 4 + 2];
                                child.w = (double) SWC[vindex.x * 4 + 3];

                                // extract parent values
                                parent.x = (double) SWC[vindex.y * 4 + 0];
                                parent.y = (double) SWC[vindex.y * 4 + 1];
                                parent.z = (double) SWC[vindex.y * 4 + 2];
                                parent.w = (double) SWC[vindex.y * 4 + 3];

                                //distance squared between child parent
                                dist2 = ((parent.x - child.x) * (parent.x - child.x)) +
                                               ((parent.y - child.y) * (parent.y - child.y)) +
                                               ((parent.z - child.z) * (parent.z - child.z));

                                // determine whether particle is inside this connection
                                bool inside = swc2v(nextpos, child, parent, dist2);

                                // if it is inside the connection we don't need to check the remaining.
                                if (inside) {
                                    // update the particles state
                                    parstate.y = 1;
                                    // end for p loop
                                    p = pairmax;
                                }
                            }
                                // if the value of the index array is -1 we have checked all pairs for this particle.
                            else {
                                // end for p loop
                                p = pairmax;
                            }
                        }
                    }
                }
                // determine if step executes
                completes = xi.w < permprob;
                // printf("Completes [step,particle] (%d, %d): %d\n",i,gid,completes);

                if (parstate.y) {
                    A[gx.x] = nextpos.x;
                    A[gx.y] = nextpos.y;
                    A[gx.z] = nextpos.z;
                } else {
                    if (completes && parstate.x) {
                        printf("ESCAPEE ALERT!\n");
                        A[gx.x] = nextpos.x;
                        A[gx.y] = nextpos.y;
                        A[gx.z] = nextpos.z;
                    }
                    if (!completes && parstate.x && !parstate.y) {
                        // printf("REJECTION ALERT!\n");
                        flag = true;
                    }
                }
            } else {
                flag = false;
            }


            dx = (A[gx.x] - xnot.x) * res;
            dy = (A[gx.y] - xnot.y) * res;
            dz = (A[gx.z] - xnot.z) * res;

            atomAdd(&dx2[6 * i + 0], dx * dx);
            atomAdd(&dx2[6 * i + 1], dx * dy);
            atomAdd(&dx2[6 * i + 2], dx * dz);
            atomAdd(&dx2[6 * i + 3], dy * dy);
            atomAdd(&dx2[6 * i + 4], dy * dz);
            atomAdd(&dx2[6 * i + 5], dz * dz);
        }
    }
}


void setupSimulation() {
    std::string path = "/autofs/homes/009/bs244/cuda-workspace/hellocuda/cudaprep/data";
    simreader reader(&path);
    simulation sim(reader);
    sim.setStep_num(1000);
    sim.setParticle_num(10000);
    double *stepsize = sim.getStep_size();
}

void inithostlut(int *hostLookup, int prod, int bx, int by, int bz) {
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

    int size = 100000;
    int iter = 100000;
    int block_size = 128;
    int NO_BYTES = size * sizeof(double);
    dim3 block(block_size);
    dim3 grid((size / block.x) + 1);
    int random_bytes = size * sizeof(double) * 3;

    double boundx = 20.0;
    double boundy = 20.0;
    double boundz = 20.0;

    int bx = (int) (boundx);
    int by = (int) (boundy);
    int bz = (int) (boundz);
    int prod = (int) (boundx * boundy * boundz);

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
    int *hostIndexArray;
    double *hostSWC;
    int indexsize = pairmax * npair;
    double *hostdx2;

    // Alloc Memory for Host Pointers
    h_a = (double *) malloc(random_bytes);
    hostBounds = (double *) malloc(3 * sizeof(double));
    hostLogicalVector = (bool *) malloc(size * sizeof(bool));
    hostLookup = (int *) malloc(prod * sizeof(int));
    hostIndexArray = (int *) malloc(pairmax * 2 * npair * sizeof(int));
    hostSWC = (double *) malloc(4 * indexsize * sizeof(double));
    hostdx2 = (double *) malloc(6 * iter * sizeof(double));


    // Set Values for Host
    memset(hostIndexArray, 0, pairmax * 2 * npair * sizeof(int));
    memset(hostSWC, 0, indexsize * sizeof(double));
    memset(hostdx2, 0, 6 * iter * sizeof(double));

    for (int i = 0; i < npair; i++) {
        printf("HINDEX@I: ");
        for (int j = 0; j < pairmax; j++) {
            hostIndexArray[pairmax * i + 2 * j + 0] = pairmax * i + j + 0;
            hostIndexArray[pairmax * i + 2 * j + 1] = pairmax * i + j + 1;
            printf("[%d,%d] ", hostIndexArray[pairmax * i + 2 * j + 0], hostIndexArray[pairmax * i + 2 * j + 1]);
        }
        printf("\n");
    }
    double counter = 0.8;
    for (int i = 0; i < indexsize; i++) {
        hostSWC[4 * i + 0] = i + counter;
        hostSWC[4 * i + 1] = -i + counter;
        hostSWC[4 * i + 2] = ((double) i / 2.0) + counter;
        hostSWC[4 * i + 3] = ((double) i) + 3.0;
        if (counter >= 10.0) {
            counter = -10.2;
        } else {
            counter += 2.900213466;
        }
    }

    memset(hostLogicalVector, false, size * sizeof(bool));
    hostBounds[0] = boundx;
    hostBounds[1] = boundy;
    hostBounds[2] = boundz;
    inithostlut(hostLookup, prod, bx, by, bz);

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
    double *devicedx2;
    clock_t start = clock();
    // Allocate Memory on Device
    cudaMalloc((int **) &deviceLookup, prod * sizeof(int));
    cudaMalloc((int **) &deviceIndexArray, pairmax * 2 * npair * sizeof(int));
    cudaMalloc((double **) &deviceSWC, 4 * indexsize * sizeof(double));
    cudaMalloc((double **) &d_a, random_bytes);
    cudaMalloc((double **) &deviceBounds, 3 * sizeof(double));
    cudaMalloc((bool **) &deviceLogicalVector, size * sizeof(bool));
    cudaMalloc((curandStatePhilox4_32_10_t * *) & deviceState, size * sizeof(curandStatePhilox4_32_10_t));
    cudaMalloc((double **) &devicedx2, 6 * iter * sizeof(double));

    // Set Values for Device
    cudaMemcpy(deviceBounds, hostBounds, 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceLookup, hostLookup, prod * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceLogicalVector, hostLogicalVector, size * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceIndexArray, hostIndexArray, pairmax * 2 * npair * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceSWC, hostSWC, 4 * indexsize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devicedx2, hostdx2, 6 * iter * sizeof(double), cudaMemcpyHostToDevice);

    /**
     * Initalize Random Stream
     */
    setup_kernel<<<grid, block>>>(deviceState, 1);

    // option for printing in kernel
    bool debug = false;

    /**
     * Call Kernel
    */

    simulate<<<grid, block>>>(d_a, devicedx2, deviceBounds, deviceLookup, deviceIndexArray, deviceSWC,
                              deviceLogicalVector, deviceState, size, pairmax, iter, debug);
    // Wait for results
    cudaDeviceSynchronize();



    /**
     * Copy Results From Device to Host
     */
    cudaMemcpy(h_a, d_a, random_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostLogicalVector, deviceLogicalVector, size * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostdx2, devicedx2, 6 * iter * sizeof(double), cudaMemcpyDeviceToHost);
    clock_t end = clock();
    double gpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("kernel took %f seconds\n", gpu_time_used);
    for (int i = 0; i < size; i++) {
        double x = h_a[3 * i + 0];
        double y = h_a[3 * i + 1];
        double z = h_a[3 * i + 2];
        // printf("x: %.4f \t y: %.4f \t z: %.4f\t Has Lookup Values: %d\n", x, y, z, hostLogicalVector[i]);
    }

    for (int i = 0; i < iter; i += 10) {
        printf("xx: %.4f \t xy: %.4f \t xz: %.4f\t yy: %.4f\t yz: %.4f\t zz: %.4f\t \n", hostdx2[i * 6 + 0],
               hostdx2[i * 6 + 1], hostdx2[i * 6 + 2], hostdx2[i * 6 + 3], hostdx2[i * 6 + 4], hostdx2[i * 6 + 5]);
    }

    /**
     * Free Device Data
     */
    cudaFree(d_a);
    cudaFree(deviceBounds);
    cudaFree(deviceState);
    cudaFree(deviceLookup);
    cudaFree(deviceLogicalVector);
    cudaFree(deviceIndexArray);
    cudaFree(devicedx2);
    cudaFree(deviceSWC);



    /**
     * Free Host Data
     */
    free(h_a);
    free(hostBounds);
    free(hostLookup);
    free(hostLogicalVector);
    return 0;
}
