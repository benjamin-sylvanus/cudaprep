#include <iostream>
#include "src/simreader.h"
#include "src/simulation.h"
#include "src/particle.h"
#include "vector"
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

using std::cout;
using std::endl;

__global__ void setup_kernel(curandStatePhilox4_32_10_t *state, unsigned long seed) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void
doaflip(double *A, double *Bounds, int *LUT, bool *conditionalExample, curandStatePhilox4_32_10_t *state, int size,
        bool debug) {

    int gid = threadIdx.x + blockDim.x * blockIdx.x;

    if (gid < size) {
        curandStatePhilox4_32_10_t localstate = state[gid];
        double4 xi;
        xi = curand_uniform4_double(&localstate);
        A[3 * gid + 0] = xi.x * Bounds[0];
        A[3 * gid + 1] = xi.y * Bounds[1];
        A[3 * gid + 2] = xi.z * Bounds[2];

        int _A[3];
        _A[0] = (int) A[3 * gid + 0];
        _A[1] = (int) A[3 * gid + 1];
        _A[2] = (int) A[3 * gid + 2];

        int b[3];
        b[0] = (int) Bounds[0];
        b[1] = (int) Bounds[1];
        b[2] = (int) Bounds[2];

        //int linearIndex = offset + bz*(by*i + j) + k;
        int linearIndex = 0 + b[2] * (b[1] * _A[0] + _A[1]) + _A[2];
        conditionalExample[gid] = (bool) LUT[linearIndex];

        if (debug)
        {
            printf("Xi: [%f %f %f]\n", xi.x, xi.y, xi.z);
            printf("Xi*b: [%f %f %f]\n", xi.x * Bounds[0], xi.y * Bounds[1], xi.z * Bounds[2]);
            printf("Floor of A: [%d %d %d]\n", _A[0], _A[1], _A[2]);
            printf("Linear Index: %d\t Value of Lookup: %d \n", linearIndex, LUT[linearIndex]);
            printf("conditionalExample[gid]: %d\n", conditionalExample[gid]);
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
    particle par(stepsize);
    par.display();
    par.setFlag();
    par.setState();
    double position[3] = {1, 2, 3};
    par.setPos(position);
    par.setState(false);
    par.display();

    sim.setParticle_num(100);
    double poses[int(3 * sim.getParticle_num())];
    double *nextPositions = sim.nextPosition(poses);
    double nextvectors[int(3 * sim.getParticle_num())];
    double *coords;
    auto elapsed = clock();
    time_t time_req;
    time_req = clock();
    for (int i = 0; i < 10; i++) {
        time_req = clock();
        coords = sim.nextPosition(nextvectors);
        for (int j = 0; j < int(3 * sim.getParticle_num()); j += 3) {
            printf("Sum:\n%.3f + %.3f | %.3f\n%.3f + %.3f | %.3f\n%.3f + %.3f | %.3f\n\n", nextPositions[j + 0],
                   coords[j + 0], nextPositions[j + 0] + coords[j + 0], nextPositions[j + 1], coords[j + 1],
                   nextPositions[j + 1] + coords[j + 1], nextPositions[j + 2], coords[j + 2],
                   nextPositions[j + 2] + coords[j + 2]);
            nextPositions[j + 0] = nextPositions[j + 0] + coords[j + 0];
            nextPositions[j + 1] = nextPositions[j + 1] + coords[j + 1];
            nextPositions[j + 2] = nextPositions[j + 2] + coords[j + 2];
            printf("Position After:\t[%.3f, %.3f, %.3f]\n", nextPositions[j + 0], nextPositions[j + 1],
                   nextPositions[j + 2]);
        }
        time_req = clock() - time_req;
        std::cout << std::endl << (float) time_req / CLOCKS_PER_SEC << " seconds" << std::endl;
    }
}

void inithostlut(int * hostLookup, int prod, int bx, int by, int bz)
{
    memset(hostLookup, 0, prod * sizeof(int));
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
    hostLookup[id0] = 1;
    hostLookup[id1] = 1;
    hostLookup[id2] = 1;
    hostLookup[id3] = 1;
    hostLookup[id4] = 1;
    hostLookup[id01] = 1;
    hostLookup[id11] = 1;
    hostLookup[id21] = 1;
    hostLookup[id31] = 1;
    hostLookup[id41] = 1;
}

int main() {

    setupSimulation();
    /**
     * Read Simulation and Initialize Object
     */

    std::string path = "/autofs/homes/009/bs244/cuda-workspace/hellocuda/cudaprep/data";
    simreader reader(&path);
    simulation sim(reader);
    sim.setStep_num(1000);
    sim.setParticle_num(10000);

    std::vector<double> simulationparams = sim.getParameterdata();

    for (double d:simulationparams)
    {
        printf("%f\n",d);
    }

    int size = 100;
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

    // Alloc Memory for Host Pointers
    h_a = (double *) malloc(random_bytes);
    hostBounds = (double *) malloc(3 * sizeof(double));
    hostLogicalVector = (bool *) malloc(size * sizeof(bool));
    hostLookup = (int *) malloc(prod * sizeof(int));

    // Set Values for Host
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

    // Allocate Memory on Device
    cudaMalloc((int **) &deviceLookup, prod * sizeof(int));
    cudaMalloc((double **) &d_a, random_bytes);
    cudaMalloc((double **) &deviceBounds, 3 * sizeof(double));
    cudaMalloc((bool **) &deviceLogicalVector, size * sizeof(bool));
    cudaMalloc((curandStatePhilox4_32_10_t * *) & deviceState, size * sizeof(curandStatePhilox4_32_10_t));

    // Set Values for Device
    cudaMemcpy(deviceBounds, hostBounds, 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceLookup, hostLookup, prod * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceLogicalVector, hostLogicalVector, size * sizeof(bool), cudaMemcpyHostToDevice);

    /**
     * Initalize Random Stream
     */
    setup_kernel<<<grid, block>>>(deviceState, 1);

    // option for printing in kernel
    bool debug = false;

    /**
     * Call Kernel
     */
    doaflip<<<grid, block>>>(d_a, deviceBounds, deviceLookup, deviceLogicalVector, deviceState, size, debug);

    // Wait for results
    cudaDeviceSynchronize();

    /**
     * Copy Results From Device to Host
     */
    cudaMemcpy(h_a, d_a, random_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostLogicalVector, deviceLogicalVector, size * sizeof(bool), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++) {
        double x = h_a[3 * i + 0];
        double y = h_a[3 * i + 1];
        double z = h_a[3 * i + 2];
        printf("x: %.4f \t y: %.4f \t z: %.4f\t Has Lookup Values: %d\n", x, y, z, hostLogicalVector[i]);
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
