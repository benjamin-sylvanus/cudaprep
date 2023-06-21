#include "./src/simreader.h"
#include "./src/simulation.h"
#include "./src/controller.h"
#include "./src/viewer.h"
#include "./src/funcs.h"
#include "./src/overloads.h"
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
#include <chrono>
#include <thread>
#include <cublas_v2.h>

#define CUDART_PI_F 3.141592654f
#define PI 3.14159265358979323846
#define LDPI 3.141592653589793238462643383279502884L
#define timepoints 1000
#define Nc 2
#define Nbvec 3
#define SOD (sizeof(double))
#define SOI (sizeof(int))
#define SOF (sizeof(float))
#define SOD3 (sizeof(double3))
#define SOD4 (sizeof(double4))
#define SOI3 (sizeof(int3))
#define SOI4 (sizeof(int4))

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

using std::cout;
using std::cin;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


/**
 * @brief Initializes the random number generator
 * @param state pointer to the random number generator
 * @param seed seed for the random number generator
 */
__global__ void setup_kernel(curandStatePhilox4_32_10_t *state, unsigned long seed) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}


/**
 * @brief Simulation Kernel for the GPU
 * @param savedata - the data to be saved
 * @param dx2 - the second moment of the diffusion tensor
 * @param dx4 - the fourth moment of the diffusion tensor
 * @param Bounds - the bounds of the simulation
 * @param state - the random number generator state
 * @param SimulationParams - the simulation parameters
 * @param d4swc - the swc data
 * @param nlut - the neighbor lookup table
 * @param NewIndex - the new index
 * @param IndexSize - the index size
 * @param size - the number of particles
 * @param iter - the number of iterations
 * @param debug - whether or not to print debug statements
 * @param point - the point to simulate
 * @param SaveAll - whether or not to save all data
 * @param Reflections - the reflections
 * @param Uref - the unreflected data
 * @param flip - the flip data
 * @param T2 - the T2 data
 * @param T - the T data
 * @param Sig0 - the Sig0 data
 * @param SigRe - the SigRe data
 * @param BVec - the BVec data
 * @param BVal - the BVal data
 * @param TD - the TD data
 */
__global__ void simulate(double *savedata, double *dx2, double *dx4, int3 Bounds, curandStatePhilox4_32_10_t *state,
                         double *SimulationParams,
                         double4 *d4swc, int *nlut, int *NewIndex, int3 IndexSize, int size, int iter, bool debug,
                         double3 point, int SaveAll, double * Reflections, double * Uref, int * flip,
                         double * T2, double * T, double * Sig0, double * SigRe, double* BVec, double * BVal, double * TD) {

    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gid < size) {
        /**
            @index particle_num = SimulationParams[0]
            @index step_num = SimulationParams[1]
            @index step_size = SimulationParams[2]
            @index perm_prob = SimulationParams[3]
            @index init_in = SimulationParams[4]
            @index D0 = SimulationParams[5]
            @index d = SimulationParams[6]
            @index scale = SimulationParams[7]
            @index tstep = SimulationParams[8]
        */
        double step_size = SimulationParams[2];
        double perm_prob = SimulationParams[3];
        int init_in = (int) SimulationParams[4];
        double tstep = SimulationParams[8];
        double vsize = SimulationParams[9];
        double3 A;
        int2 parstate;
        double4 xi;
        double3 nextpos;
        double3 xnot;
        int3 upper;
        int3 lower;
        int3 floorpos;
        int Tstep=iter/timepoints;
        double fstep = 1;

        int3 b_int3 = make_int3(Bounds.x, Bounds.y, Bounds.z);
        int3 i_int3 = make_int3(IndexSize.x, IndexSize.y, IndexSize.z);
    
        double _T2[Nc]; 
        double t2[3] = {80, 40, 60};
        for (int j = 0; j < Nc; j++)
        {
            _T2[j] = t2[j];
        }

        double3 d2 = make_double3(0.0, 0.0, 0.0);
        bool completes;
        bool flag;
        double step = step_size;

        // init local state var
        curandStatePhilox4_32_10_t localstate = state[gid];
        xi = curand_uniform4_double(&localstate);

        // initialize position inside cell
        A = initPosition(gid, dx2, Bounds, state, SimulationParams, d4swc, nlut, NewIndex, IndexSize,
                         size, iter, init_in, debug, point);

        // record initial position
        xnot = make_double3(A.x, A.y, A.z);

        // flag is initially false
        flag = false;

        // state is based on intialization conditions if particles are required to start inside then parstate -> [1,1]
        parstate = make_int2(1, 1);

        // parlut defines whether particle is within bounds of LUT
        int parlut = 1;
        double t[Nc] ={0}; // add tstep for step in compartment

        // iterate over steps
        for (int i = 0; i < iter; i++) {
                // generate uniform randoms for step
                xi = curand_uniform4_double(&localstate);

                // determine if step executes
                completes = xi.w < perm_prob;

                // compute next position
                double pi = PI;
                computeNext(A, step, xi, nextpos, pi);

                // check coordinate validity
                validCoord(nextpos, A, b_int3, upper, lower, floorpos, Reflections, Uref, gid, i, size, iter, flip);

                // floor of next position -> check voxels
                floorpos = make_int3((int) nextpos.x, (int) nextpos.y, (int) nextpos.z);

                // extract lookup table value
                int test_lutvalue = nlut[s2i(floorpos,b_int3)];

                // for each connection check if particle inside
                bool inside = checkConnections(i_int3, test_lutvalue, nextpos, NewIndex, d4swc, fstep);

                parstate.y = (inside) ? 1 : 0;

                /**
                * @cases particle inside? 0 0 - update
                * @cases particle inside? 0 1 - check if updates
                * @cases particle inside? 1 0 - check if updates
                * @cases particle inside? 1 1 - update
                */

                // particle inside: [0 0] || [1 1]
                if (parstate.x == parstate.y) { 
                    A = nextpos; 
                    if (parstate.x) {
                        t[0] = t[0] + tstep;
                    } else {
                        t[1] = t[1] + tstep;
                    }
                }

                // particle inside: [1 0]
                if (parstate.x && !parstate.y) {
                    if (completes == true) {
                        A = nextpos;
                        parstate.x = parstate.y;
                        t[0] = t[0] + tstep * fstep;
                        t[1] = t[1] + tstep * (1 - fstep);
                    } else {
                        t[0] = t[0] + tstep;
                    }
                }

                // particle inside [0 1]
                if (!parstate.x && parstate.y) {
                    if (completes == true) {
                        A = nextpos;
                        parstate.x = parstate.y;
                        t[0] = t[0] + tstep * 1-fstep;
                        t[1] = t[1] + tstep * fstep;
                    } else {
                        t[1] = t[1] + tstep;
                    }
                }

            // Store Position Data
            if (SaveAll) {
                int3 dix = make_int3(size, iter, 3);
                int3 did[4];
                did[0] = make_int3(gid, i, 0);
                did[1] = make_int3(gid, i, 1);
                did[2] = make_int3(gid, i, 2);
                did[3] = make_int3(s2i(did[0], dix), s2i(did[1], dix), s2i(did[2], dix));
                set(savedata, did[3], A);
            }
            // Store Tensor Data
            {
                diffusionTensor(&A, &xnot, vsize, dx2, dx4, &d2, i, gid, iter, size);
                // https://github.com/NYU-DiffusionMRI/monte-carlo-simulation-3D-RMS/blob/master/part1_demo3_simulation.m

            }

            // Store Signal Data
            {

                if (i%Tstep == 0)
                {
                    int tidx=i/Tstep;
                    // loop over compartments
                    double s0 = 0.0;
                    for (int j = 0; j < 2; j++) {
                        /**
                            * @var s0 is our summation variable
                            * @var t[j] is the time in compartment j
                            * @var T2 is the T2 Relaxation in Compartment j
                        */  
                            s0 = s0 + (double) (t[j] / _T2[j]); // TODO implement "t" as time in each compartment

                    }


                    s0 = exp(-1.0 * s0);
                    atomicAdd(&Sig0[tidx],s0);
                    for (int j = 0; j < Nc; j++)
                    {
                        t[j]= 0;
                    }
                }
                // Signal
        

                /*{
                    // loop over b values
                    for (int j = 0; j < Nbvec; j++) {
                        // bval is the b value
                        // bvec is the gradient direction vector
                        // TD is the diffusion time
                        // qx is the q value
                        // d2.x, d2.y, d2.z are the displacements defined as fabs((A - xnot) * vsize);
                        // get the bvec for the current b value
                        bd = make_double3(bvec[j * 3 + 0], bvec[j * 3 + 1], bvec[j * 3 + 2]);
                        bv = bval[j];
                        td = TD[tidx];
                        // qx = sqrt(bv / td) * dot(d2, bd);
                        qx = sqrt(bv / td) * (d2.x * bd.x + d2.y * bd.y + d2.z * bd.z);
                        qx = sqrt(bval[j] / TD[tidx]) * (d2.x * bvec[j * 3 + 0] + d2.y * bvec[j * 3 + 1] + d2.z * bvec[j * 3 + 2]);
                        // qx = sqrt(bval[j] / TD[tidx]) * (dx * bvec[j * 3 + 0] + dy * bvec[j * 3 + 1] + dz * bvec[j * 3 + 2]);
                        atomAdd(&sigRe[Nbvec * tidx + j], s0 * cos(qx));
                    }
                }*/
            }
        }
    }
}

int main(int argc, char *argv[]) {
    cudaEvent_t start_c, stop_c;
    cudaEventCreate(&start_c);
    cudaEventCreate(&stop_c);
    float milliseconds = 0;
    system("clear");
    int size = 10;
    int iter = 10;
    int SaveAll;
    std::string path;
    controller control;

    /**
     * Read Simulation and Initialize Object
     */
    // Parse Arguments
    if (argc < 2) {
        path = "/autofs/space/symphony_002/users/BenSylvanus/cuda/Sims";
        std::string InPath = path;
        std::string OutPath = path;
        InPath.append("/data");
        OutPath.append("/results");
        control.Setup(InPath, OutPath, 0);
        control.start();
    } else {
        control.Setup(argc, argv, 1);
    }

    system("clear");

    double simparam[10];
    simulation sim = control.getSim();
    printf("Path: %s\n", sim.getResultPath().c_str());
    path = sim.getResultPath();
    size = sim.getParticle_num();
    iter = sim.getStep_num();
    std::vector<double> simulationparams = sim.getParameterdata();
    SaveAll = sim.getSaveAll();

    for (int i = 0; i < 10; i++) {
        double value = simulationparams[i];
        simparam[i] = value;
    }

    int block_size = 256;
    dim3 block(block_size);
    dim3 grid((size / block.x) + 1);

    std::vector <uint64_t> bounds = sim.getbounds();
    int boundx = (int) bounds[0]; int boundy = (int) bounds[1]; int boundz = (int) bounds[2];
    int prod = (int) (boundx * boundy * boundz);
    std::vector<double> r_swc = sim.getSwc();
    int nrow = r_swc.size() / 6;

    double4 swc_trim[nrow];
    double w_swc[nrow * 4];
    for (int i = 0; i < nrow; i++) {
        swc_trim[i].x = r_swc[i + nrow * 1];
        swc_trim[i].y = r_swc[i + nrow * 2];
        swc_trim[i].z = r_swc[i + nrow * 3];
        swc_trim[i].w = r_swc[i + nrow * 4];
    }
    for (int i = 0; i < nrow; i++) {
        w_swc[4 * i + 0] = r_swc[i + nrow * 1];
        w_swc[4 * i + 1] = r_swc[i + nrow * 2];
        w_swc[4 * i + 2] = r_swc[i + nrow * 3];
        w_swc[4 * i + 3] = r_swc[i + nrow * 4];
    }

    std::vector <uint64_t> lut = sim.getLut();
    std::vector <uint64_t> indexarr = sim.getIndex();
    std::vector <std::vector<uint64_t>> arrdims = sim.getArraydims();
    std::vector <uint64_t> swc_dims = arrdims[0];
    std::vector <uint64_t> lut_dims = arrdims[1];
    std::vector <uint64_t> index_dims = arrdims[2];
    std::vector <uint64_t> pairs_dims = arrdims[3];
    std::vector <uint64_t> bounds_dims = arrdims[4];
    int newindexsize = index_dims[0] * index_dims[1] * index_dims[2];

    ///Host Section

    // Create Host Pointers
    double *hostdx2;
    double *hostdx4;
    double *hostSimP;
    int *hostNewLut;
    int *hostNewIndex;
    double4 *hostD4Swc;
    double *mdx2;
    double *mdx4;
    double *hostAllData;
    double *hostReflections;
    double *hosturef;
    int* hostFlip;
    double *hostT2; // Nc * 1
    double *hostT; // Nc * 1
    double *hostSigRe; // Nbvec * iter
    double *hostSig0; // Nc * iter
    double *hostbvec; // Nbvec * 3 (x,y,z)
    double *hostbval; // Nbvec * 1 (b)
    double *hostTD;   // TD is the time elapsed at timepoint i.

    // Alloc Memory for Host Pointers
    {

        hostdx2 = (double *) malloc(6 * iter * SOD);
        hostdx4 = (double *) malloc(15 * iter * SOD);
        hostSimP = (double *) malloc(10 * SOD);
        hostD4Swc = (double4 *) malloc(nrow * SOD4);
        hostNewLut = (int *) malloc(prod * SOI);
        hostNewIndex = (int *) malloc(newindexsize * SOI);
        mdx2 = (double *) malloc(6 * iter * SOD);
        mdx4 = (double *) malloc(15 * iter * SOD);
        if (SaveAll) {
            hostAllData = (double *) malloc(3 * iter * size * SOD);
        } else {
            hostAllData = (double *) malloc(3 * SOD);
        }
        hostReflections = (double *) malloc(3 *iter * size * SOD);
        hosturef = (double *) malloc(3 *iter * size * SOD);
        hostFlip = (int *) malloc(3 * size * SOI);

        // Signal Variables
        hostT2 = (double *) malloc(Nc * SOD);
        hostT = (double *) malloc(Nc * SOD);
        hostSigRe = (double *) malloc(Nbvec * timepoints * SOD);
        hostSig0 = (double *) malloc(timepoints * SOD);
        hostbvec = (double *) malloc(Nbvec * 3 * SOD);
        hostbval = (double *) malloc(Nbvec * SOD);
        hostTD = (double *) malloc(Nbvec * SOD);
        printf("Allocated Host Data\n");
    }

    // Set Values for Host
    {
        memset(hostdx2, 0.0, 6 * iter * SOD);
        memset(hostdx4, 0.0, 15 * iter * SOD);
        {

            for (int i = 0; i < 10; i++) {
                hostSimP[i] = simparam[i];
            }

            for (int i = 0; i < nrow; i++) {
                hostD4Swc[i].x = swc_trim[i].x;
                hostD4Swc[i].y = swc_trim[i].y;
                hostD4Swc[i].z = swc_trim[i].z;
                hostD4Swc[i].w = swc_trim[i].w;
            }

            for (int i = 0; i < prod; i++) {
                int value = lut[i];
                hostNewLut[i] = value;
            }

            for (int i = 0; i < indexarr.size(); i++) {
                int value = indexarr[i];
                hostNewIndex[i] = value;
            }
        }
        memset(mdx2, 0.0, 6 * iter * SOD);
        memset(mdx4, 0.0, 15 * iter * SOD);

        if (SaveAll) {
            memset(hostAllData, 0.0, 3 * iter * size * SOD);
        } else {
            memset(hostAllData, 0.0, 3 * SOD);
        }
        memset(hostReflections, 0.0, 3 * iter * size * SOD);
        memset(hosturef, 0.0, 3 * iter * size * SOD);
        memset(hostFlip, 0.0, 3 * size * SOI);

        // signal variables
        memset(hostT2, 0.0, Nc * SOD); // T2 is read from file?
        memset(hostT, 0.0, Nc * SOD); // T is set to 0.0
        memset(hostSigRe, 0.0, Nbvec * timepoints * SOD); // Calculated in kernel
        memset(hostSig0, 0.0, timepoints * SOD); // Calculated in kernel
        memset(hostbvec, 0.0, Nbvec * 3 * SOD); // bvec is read from file
        memset(hostbval, 0.0, Nbvec * SOD); // bval is read from file
        memset(hostTD, 0.0, Nbvec * SOD); // TD is read from file
        printf("Set Host Values\n");
    }

    /**
     * Device Section:
     * - Create Pointers
     * - Allocate Memory
     * - Set Values
     */
    // Create Device Pointers
    curandStatePhilox4_32_10_t *deviceState;
    double *devicedx2;
    double *devicedx4;
    double *deviceSimP;
    double4 *deviced4Swc;
    int *deviceNewLut;
    int *deviceNewIndex;
    double *deviceAllData;
    double *deviceReflections;
    double *deviceURef;
    int *deviceFlip;

    // signal variables
    double *deviceT2;
    double *deviceT;
    double *deviceSigRe;
    double *deviceSig0;
    double *devicebvec;
    double *devicebval;
    double *deviceTD;


    clock_t start = clock();
    cudaEventRecord(start_c);

    // Allocate Memory on Device
    {

        gpuErrchk(cudaMalloc((double **) &devicedx2, 6 * iter * SOD));
        gpuErrchk(cudaMalloc((double **) &devicedx4, 15 * iter * SOD));
        gpuErrchk(cudaMalloc((curandStatePhilox4_32_10_t * *) & deviceState, size * sizeof(curandStatePhilox4_32_10_t)));
        gpuErrchk(cudaMalloc((double **) &deviceSimP, 10 * SOD));
        gpuErrchk(cudaMalloc((double4 * *) & deviced4Swc, nrow * SOD4));
        gpuErrchk(cudaMalloc((int **) &deviceNewLut, prod * SOI));
        gpuErrchk(cudaMalloc((int **) &deviceNewIndex, newindexsize * SOI));
        if (SaveAll) {
            gpuErrchk(cudaMalloc((double **) &deviceAllData, 3 * iter * size * SOD));
        } else {
            gpuErrchk(cudaMalloc((double **) &deviceAllData, 3 * SOD));
        }
        gpuErrchk(cudaMalloc((double **) &deviceReflections, 3 * iter * size * SOD));
        gpuErrchk(cudaMalloc((double **) &deviceURef, 3 * iter * size * SOD));
        gpuErrchk(cudaMalloc((int **) &deviceFlip, 3 * size * SOI));

        // signal variables
        gpuErrchk(cudaMalloc((double **) &deviceT2, Nc * SOD));
        gpuErrchk(cudaMalloc((double **) &deviceT, Nc * SOD));
        gpuErrchk(cudaMalloc((double **) &deviceSigRe, Nbvec * timepoints * SOD));
        gpuErrchk(cudaMalloc((double **) &deviceSig0, timepoints * SOD));
        gpuErrchk(cudaMalloc((double **) &devicebvec, Nbvec * 3 * SOD));
        gpuErrchk(cudaMalloc((double **) &devicebval, Nbvec * SOD));
        gpuErrchk(cudaMalloc((double **) &deviceTD, Nbvec * SOD));
        printf("Device Memory Allocated\n");
    }

    // Set Values for Device
    {
        printf("Copying Host data to Device\n");
        gpuErrchk(cudaMemcpy(devicedx2, hostdx2, 6 * iter * SOD, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(devicedx4, hostdx4, 15 * iter * SOD, cudaMemcpyHostToDevice));
        setup_kernel<<<grid, block>>>(deviceState, 1); // initialize the random states
        gpuErrchk(cudaMemcpy(deviceSimP, hostSimP, 10 * SOD, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(deviced4Swc, hostD4Swc, nrow * SOD4, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(deviceNewLut, hostNewLut, prod * SOI, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(deviceNewIndex, hostNewIndex, newindexsize * SOI, cudaMemcpyHostToDevice));
        if (SaveAll) {
            gpuErrchk(cudaMemcpy(deviceAllData, hostAllData, 3 * iter * size * SOD, cudaMemcpyHostToDevice));
        } else {
            gpuErrchk(cudaMemcpy(deviceAllData, hostAllData, 3 * SOD, cudaMemcpyHostToDevice));
        }
        gpuErrchk(cudaMemcpy(deviceReflections, hostReflections, 3 * iter * size * SOD, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(deviceURef, hosturef, 3 * iter * size * SOD, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(deviceFlip, hostFlip, 3 * size * SOI, cudaMemcpyHostToDevice));

        // signal variables
        gpuErrchk(cudaMemcpy(deviceT2, hostT2, Nc * SOD, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(deviceT, hostT, Nc * SOD, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(deviceSigRe, hostSigRe, Nbvec * timepoints * SOD, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(deviceSig0, hostSig0, timepoints * SOD, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(devicebvec, hostbvec, Nbvec * 3 * SOD, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(devicebval, hostbval, Nbvec * SOD, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(deviceTD, hostTD, Nbvec * SOD, cudaMemcpyHostToDevice));

    }

    // option for printing in kernel
    bool debug = false;
    double3 point = make_double3(hostD4Swc[0].x, hostD4Swc[0].y, hostD4Swc[0].z);
    int3 deviceBounds = make_int3(boundx, boundy, boundz);
    int3 deviceIndexSize = make_int3(index_dims[0], index_dims[1], index_dims[2]);



    /**
     * Call Kernel
    */


    // kernel
    {
        printf("Simulating...\n");
        simulate<<<grid, block>>>(deviceAllData, devicedx2, devicedx4, deviceBounds, deviceState, deviceSimP,
                                  deviced4Swc, deviceNewLut, deviceNewIndex, deviceIndexSize, size, iter, debug, point,
                                  SaveAll,
                                  deviceReflections, deviceURef, deviceFlip, // reflection variables
                                  deviceT2, deviceT, deviceSig0, deviceSigRe, devicebvec, devicebval, deviceTD); // signal variables
        cudaEventRecord(stop_c);
    }

    // Wait for results
    cudaDeviceSynchronize();

    clock_t end = clock();
    double gpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Simulation took %f seconds\n", gpu_time_used);

    /**
     * Copy Results From Device to Host
     */
    printf("Copying back to Host\n");

    // cudaMemcpyDeviceToHost
    {

        gpuErrchk(cudaMemcpy(hostdx2, devicedx2, 6 * iter * SOD, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(hostdx4, devicedx4, 15 * iter * SOD, cudaMemcpyDeviceToHost));
        if (SaveAll) {
            gpuErrchk(cudaMemcpy(hostAllData, deviceAllData, 3 * iter * size * SOD, cudaMemcpyDeviceToHost));
        } else {
            gpuErrchk(cudaMemcpy(hostAllData, deviceAllData, 3 * SOD, cudaMemcpyDeviceToHost));
        }
        // Reflection Variables
        gpuErrchk(cudaMemcpy(hostReflections, deviceReflections, 3 * iter * size * SOD, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(hosturef, deviceURef, 3 * iter * size * SOD, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(hostFlip, deviceFlip, 3 * size * SOI, cudaMemcpyDeviceToHost));

        // Signal Variables
        gpuErrchk(cudaMemcpy(hostT2, deviceT2, Nc * SOD, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(hostT, deviceT, Nc * SOD, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(hostSigRe, deviceSigRe, Nbvec * timepoints * SOD, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(hostSig0, deviceSig0, timepoints * SOD, cudaMemcpyDeviceToHost));
    }

    cudaEventSynchronize(stop_c);
    cudaEventElapsedTime(&milliseconds, start_c, stop_c);
    end = clock();
    printf("Kernel took %f seconds\n", milliseconds / 1e3);
    auto t1 = high_resolution_clock::now();

    // Free Device Memory
    {

        printf("Freeing Device Data: ");
        gpuErrchk(cudaFree(deviceState));
        gpuErrchk(cudaFree(devicedx2));
        gpuErrchk(cudaFree(devicedx4));
        gpuErrchk(cudaFree(deviceSimP));
        gpuErrchk(cudaFree(deviced4Swc));
        gpuErrchk(cudaFree(deviceNewLut));
        gpuErrchk(cudaFree(deviceNewIndex));
        gpuErrchk(cudaFree(deviceAllData));

        // Reflection Variables
        gpuErrchk(cudaFree(deviceReflections));
        gpuErrchk(cudaFree(deviceURef));
        gpuErrchk(cudaFree(deviceFlip));

        // Signal Variables
        gpuErrchk(cudaFree(deviceT2));
        gpuErrchk(cudaFree(deviceT));
        gpuErrchk(cudaFree(deviceSigRe));
        gpuErrchk(cudaFree(deviceSig0));
        gpuErrchk(cudaFree(devicebvec));
        gpuErrchk(cudaFree(devicebval));
        gpuErrchk(cudaFree(deviceTD));

    }

    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    printf("%f seconds\n", ms_double.count() / 1e3);
    printf("Writing results: ");

    // Write Results
    {

        std::string outpath = sim.getResultPath();
        t1 = high_resolution_clock::now();
        writeResults(hostdx2, hostdx4, mdx2, mdx4, hostSimP, w_swc, iter, size, nrow, outpath);
        std::string allDataPath = outpath;
        if (SaveAll) {
            allDataPath.append("/allData.bin");
            FILE *outFile = fopen(allDataPath.c_str(), "wb");
            fwrite(hostAllData, SOD, iter * size * 3, outFile);
            fclose(outFile);
        }
        // write reflections and uref
        std::string reflectionsPath = outpath;
        reflectionsPath.append("/reflections.bin");
        FILE *outFile = fopen(reflectionsPath.c_str(), "wb");
        fwrite(hostReflections, SOD, iter * size * 3, outFile);
        fclose(outFile);
        std::string urefPath = outpath;
        urefPath.append("/uref.bin");
        outFile = fopen(urefPath.c_str(), "wb");
        fwrite(hosturef, SOD, iter * size * 3, outFile);
        fclose(outFile);

        // write sig0 and sigRe
        std::string sig0Path = outpath;
        sig0Path.append("/sig0.bin");
        outFile = fopen(sig0Path.c_str(), "wb");
        fwrite(hostSig0, SOD, timepoints, outFile);
        fclose(outFile);
        std::string sigRePath = outpath;
        sigRePath.append("/sigRe.bin");
        outFile = fopen(sigRePath.c_str(), "wb");
        fwrite(hostSigRe, SOD, Nbvec * iter, outFile);
        fclose(outFile);
    }

    t2 = high_resolution_clock::now();
    ms_double = t2 - t1;
    printf("%f seconds\n", ms_double.count() / 1e3);

    // Free Host Memory
    {

        free(hostdx2);
        free(hostdx4);
        free(hostSimP);
        free(hostD4Swc);
        free(hostNewIndex);
        free(mdx2);
        free(mdx4);
        free(hostAllData);

        // Reflection Variables
        free(hostReflections);
        free(hosturef);
        free(hostFlip);

        // Signal Variables
        free(hostT2);
        free(hostT);
        free(hostSigRe);
        free(hostSig0);
        free(hostbvec);
        free(hostbval);
        free(hostTD);
    }
    printf("Done!\n");
    return 0;
}
