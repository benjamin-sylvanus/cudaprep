#include "./src/newsimreader.h"
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
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

using std::cin;
using std::cout;
using std::endl;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

/**
 * @brief Initializes the random number generator
 * @param state pointer to the random number generator
 * @param seed seed for the random number generator
 */
__global__ void setup_kernel(curandStatePhilox4_32_10_t *state, unsigned long seed)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

/**
 * @brief determine the volume fraction via random sampling over n particles
 * @param state
 * @param Bounds
 * @param d4swc
 * @param nlut
 * @param NewIndex
 * @param IndexSize
 * @param n
 * @param label
 * @param vf
 */
__global__ void volfrac(curandStatePhilox4_32_10_t *state, int3 Bounds, double4 *d4swc, int *nlut, int *NewIndex, int3 IndexSize, int n, int *label, double *vf)
{
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    int N = 10000;
    if (gid == 0)
    {
        printf("ok %d\n", n);
    }
    if (gid < N)
    {
        label[gid] = 0;
        int R = 0;
        double3 nextpos;
        int3 upper;
        int3 lower;
        int3 floorpos;
        int3 b_int3 = make_int3(Bounds.x, Bounds.y, Bounds.z);
        int3 i_int3 = make_int3(IndexSize.x, IndexSize.y, IndexSize.z);
        double3 b_d3 = make_double3((double)b_int3.x, (double)b_int3.y, (double)b_int3.z);
        curandStatePhilox4_32_10_t localstate = state[gid];
        double4 xr;
        double3 A;
        bool cont = true;

        // init local state var
        xr = curand_uniform4_double(&localstate);

        // set particle initial position
        A = make_double3(xr.x * b_d3.x, xr.y * b_d3.y, xr.z * b_d3.z);

        nextpos = A;

        // floor of position -> check voxels
        floorpos.x = (int)A.x;
        floorpos.y = (int)A.y;
        floorpos.z = (int)A.z;

        // upper bounds of lookup table
        upper.x = floorpos.x < b_int3.x;
        upper.y = floorpos.y < b_int3.y;
        upper.z = floorpos.z < b_int3.z;

        // lower bounds of lookup table
        lower.x = floorpos.x > 0;
        lower.y = floorpos.y > 0;
        lower.z = floorpos.z > 0;

        double4 parent;
        double4 child;
        int2 vindex;
        int id_test = s2i(floorpos, Bounds);
        int test_lutvalue = nlut[id_test];
        double dist2;

        // loop over connections
        for (int page = 0; page < i_int3.z; page++)
        {
            int3 c_new = make_int3(test_lutvalue, 0, page);
            int3 p_new = make_int3(test_lutvalue, 1, page);
            vindex.x = NewIndex[s2i(c_new, i_int3)] - 1;
            vindex.y = NewIndex[s2i(p_new, i_int3)] - 1;
            // if theres a index
            if ((vindex.x) != -1)
            {
                child = make_double4(d4swc[vindex.x].x, d4swc[vindex.x].y, d4swc[vindex.x].z, d4swc[vindex.x].w);
                parent = make_double4(d4swc[vindex.y].x, d4swc[vindex.y].y, d4swc[vindex.y].z, d4swc[vindex.y].w);

                // distance squared between child parent
                dist2 = distance2(parent, child);

                // determine whether particle is inside this connection
                bool inside = swc2v(nextpos, child, parent, dist2);

                // if it is inside the connection we don't need to check the remaining.
                if (inside)
                {
                    // update the particles state
                    label[gid] = 1;
                    // printf("It was inside\n");
                    // end for p loop
                    page = i_int3.z;
                }
            }
        }
    }

    __syncthreads(); // Correct synchronization function

    // sum the particles inside and calculate volume fraction
    if (gid == 0)
    {
        printf("gid %d\n", gid);

        double R = 0;
        for (int i = 0; i < N; i++)
        {
            R += (double)label[i];
        }

        printf("R: %.0f\n", R);
        printf("R/N: %.0f/%d = %.4f\n", R, N, R / (double)N);

        vf[0] = (double)R / (double)N; // Update the volume fraction using pointers
    }
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
                         double3 point, int SaveAll, double *Reflections, double *Uref, int *flip,
                         double *T2, double *T, double *Sig0, double *SigRe, double *BVec, double *BVal, double *TD)
{

    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gid < size)
    {
        /** TODO: Verify this order is still correct
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
        int init_in = (int)SimulationParams[4];
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
        int Tstep = iter / timepoints;
        double fstep = 1;
        double pi = PI;

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

        // prob to accept the sampling: init in: 4. per compartment. both <= to 1.  one should be 1
        A = initPosition(gid, dx2, Bounds, state, SimulationParams, d4swc, nlut, NewIndex, IndexSize,
                         size, iter, init_in, debug, point); // initialize position inside cell

        xnot = make_double3(A.x, A.y, A.z); // record initial position
        flag = false;                       // flag is initially false

        parstate = make_int2(1, 1); // particle state [previous step, current step]
        int parlut = 1;             // particle within bounds of LUT?
        double t[Nc] = {0};         // add tstep for step in compartment

        for (int i = 0; i < iter; i++)
        {
            xi = curand_uniform4_double(&localstate); // generate uniform randoms for step
            completes = xi.w < perm_prob;             // determine if step executes
            computeNext(A, step, xi, nextpos, pi);    // compute the next position

            // check coordinate validity
            validCoord(nextpos, A, b_int3, upper, lower, floorpos, Reflections, Uref, gid, i, size, iter, flip);
            floorpos = make_int3((int)nextpos.x, (int)nextpos.y, (int)nextpos.z);
            int test_lutvalue = nlut[s2i(floorpos, b_int3)];
            bool inside = checkConnections(i_int3, test_lutvalue, nextpos, NewIndex, d4swc, fstep); // check if particle inside
            parstate.y = (inside) ? 1 : 0;

            /**
            * @cases particle inside? 0 0 - update
            * @cases particle inside? 0 1 - check if updates
            * @cases particle inside? 1 0 - check if updates
            * @cases particle inside? 1 1 - update

            * @vars:
            * t[0]: inside
            * t[1]: outside
            */

            // particle inside: [0 0] || [1 1]
            if (parstate.x == parstate.y)
            {
                A = nextpos;
                if (parstate.x)
                {
                    t[0] = t[0] + tstep;
                }
                else
                {
                    t[1] = t[1] + tstep;
                }
            }

            // particle inside: [1 0]
            if (parstate.x && !parstate.y)
            {
                if (!completes)
                {
                    t[0] = t[0] + tstep;
                }
                else
                {
                    A = nextpos;
                    parstate.x = parstate.y;
                    t[0] = t[0] + tstep * fstep;
                    t[1] = t[1] + tstep * (1 - fstep);
                }
            }

            // particle inside [0 1]
            if (!parstate.x && parstate.y)
            {
                if (!completes)
                {
                    t[1] = t[1] + tstep;
                }
                else
                {
                    A = nextpos;
                    parstate.x = parstate.y;
                    t[0] = t[0] + tstep * (1 - fstep);
                    t[1] = t[1] + tstep * fstep;
                }
            }

            // Store Position Data
            if (SaveAll)
            {
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
                if (i % Tstep == 0)
                {
                    int tidx = i / Tstep;
                    // loop over compartments
                    double s0 = 0.0;
                    for (int j = 0; j < 2; j++)
                    {
                        /**
                         * @var s0 is our summation variable
                         * @var t[j] is the time in compartment j
                         * @var T2 is the T2 Relaxation in Compartment j
                         */
                        s0 = s0 + (double)(t[j] / _T2[j]); // TODO implement "t" as time in each compartment
                    }

                    s0 = exp(-1.0 * s0);
                    atomicAdd(&Sig0[tidx], s0);

                    // // Signal
                    // for(int j = 0; j < Nbvec; j++)
                    // {
                    //     // access b value and b vector
                    //     double bval = bvalues[j];
                    //     double3 bvec =  bvectors[j];
                    //     td = TD[tidx];
                    //     qx = sqrt(b.w/td) * dot(d2,bvec)
                    //     atomicAdd(&sigRe[Nbvec * tidx + j], s0 * cos(qx));
                    // }

                    for (int j = 0; j < Nc; j++)
                    {
                        t[j] = 0;
                    }
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{
    cudaEvent_t start_c, stop_c;
    cudaEventCreate(&start_c);
    cudaEventCreate(&stop_c);
    float milliseconds = 0;
    int NC = 2;
    system("clear");
    int SaveAll = 1;
    std::string root;
    // Parse config path from arguments
    if (argc < 2) {
        root = "data/simulation_configs/test";
    } else {
        root = argv[1];
    }

    const std::string binaryFile = root + "/simulation_config.bin";
    const std::string jsonFile = root + "/simulation_config.json";

    std::vector<Variable> variables = NewSimReader::parseJson(jsonFile);
    NewSimReader::readBinaryFile(binaryFile, variables);

    // Initialize variables to hold extracted data
    uint64_t particleNum = 0, stepNum = 0;
    double stepSize = 0, permProb = 0;
    double initIn = 0, D0 = 0, d = 0, scale = 0, tstep = 0, vsize = 0;
    double *swcmat = nullptr;
    uint64_t *LUT = nullptr;
    uint64_t *C = nullptr;
    uint64_t *pairs = nullptr;
    uint64_t *boundSize = nullptr;

    // Extract data from variables
    NewSimReader::extractData(variables, particleNum, stepNum, stepSize, permProb,
                            initIn, D0, d, scale, tstep, vsize,
                            swcmat, LUT, C, pairs, boundSize, true);
    NewSimReader::previewConfig(variables, particleNum, stepNum, stepSize, permProb, initIn, D0, d, scale, tstep, vsize,
                            swcmat, LUT, C, pairs, boundSize, true);


    // Get array dimensions from variables
    std::vector<std::vector<int>> arrdims;
    std::vector<int> swc_dims, lut_dims, index_dims, pairs_dims, bounds_dims;

    for (const auto& var : variables) {
        if (var.name == "swcmat") {
            swc_dims = var.size;           
        }
        else if (var.name == "LUT") lut_dims = var.size;
        else if (var.name == "C") index_dims = var.size;
        else if (var.name == "pairs") pairs_dims = var.size;
        else if (var.name == "bounds") bounds_dims = var.size;
    }

    arrdims = {swc_dims, lut_dims, index_dims, pairs_dims, bounds_dims};    // Calculate new index size
    int newindexsize = index_dims[0] * index_dims[1] * index_dims[2];
    int LUTSIZE = lut_dims[0] * lut_dims[1] * lut_dims[2];
    // Convert to simulation parameters
    int size = static_cast<int>(particleNum);
    int iter = static_cast<int>(stepNum);
    double simparam[10] = {
        static_cast<double>(size),
        static_cast<double>(iter),
        stepSize, permProb, initIn,
        D0, d, scale, tstep, vsize
    };

    // Get dimensions from boundSize
    int3 bounds = make_int3(boundSize[0], boundSize[1], boundSize[2]);
    int prod = bounds.x * bounds.y * bounds.z;

    // Get nrow from swcmat dimensions
    size_t nrow = swc_dims[0];  // Now we can use swc_dims directly

    // Convert swcmat to r_swc format
    std::vector<double> r_swc(nrow * 6);
    for (int i = 0; i < nrow; i++) {
        for (int j = 0; j < 6; j++) {
            r_swc[i + nrow * j] = swcmat[i * 6 + j];
        }
    }
    // Create double4 array for swc_trim
    double4 swc_trim[nrow];
    double w_swc[nrow * 4];
    for (int i = 0; i < nrow; i++) {
        swc_trim[i].x = swcmat[i + nrow * 1];
        swc_trim[i].y = swcmat[i + nrow * 2];
        swc_trim[i].z = swcmat[i + nrow * 3];
        swc_trim[i].w = swcmat[i + nrow * 4];
    }
    for (int i = 0; i < nrow; i++) {
        w_swc[4 * i + 0] = swc_trim[i].x;
        w_swc[4 * i + 1] = swc_trim[i].y;
        w_swc[4 * i + 2] = swc_trim[i].z;
        w_swc[4 * i + 3] = swc_trim[i].w;
    }
    int block_size = 256;
    dim3 block(block_size);
    dim3 grid((size / block.x) + 1);
    size_t sa_size = (SaveAll) ? 
    static_cast<size_t>(particleNum) * static_cast<size_t>(stepNum) : 1;
    std::vector<uint64_t> lut(LUT, LUT + LUTSIZE);
    std::vector<uint64_t> indexarr(C,C + newindexsize);

    /// Unified Memory
    // Declare Unified Memory Pointers
    double4 *u_D4Swc;
    double *u_dx2, *u_dx4, *u_SimP, *u_T2, *u_T, *u_SigRe, *u_Sig0, *u_bvec,
        *u_bval, *u_TD, *mdx2, *mdx4, *u_AllData, *u_Reflections, *u_uref;

    int *u_NewLut, *u_NewIndex, *u_Flip;


    // new
    int *u_label;
    double *u_vf;
    int n = 10000000;
    cudaMallocManaged(&u_D4Swc, nrow * SOD4);
    cudaMallocManaged(&u_dx2, 6 * iter * SOD);
    cudaMallocManaged(&u_dx4, 15 * iter * SOD);
    cudaMallocManaged(&u_SimP, 10 * SOD);
    cudaMallocManaged(&mdx2, 6 * iter * SOD);
    cudaMallocManaged(&mdx4, 15 * iter * SOD);
    cudaMallocManaged(&u_AllData, 3 * sa_size * SOD);
    cudaMallocManaged(&u_Reflections, 3 * iter * size * SOD);
    cudaMallocManaged(&u_uref, 3 * iter * size * SOD);
    cudaMallocManaged(&u_T2, Nc * SOD);
    cudaMallocManaged(&u_T, Nc * SOD);
    cudaMallocManaged(&u_SigRe, Nbvec * timepoints * SOD);
    cudaMallocManaged(&u_Sig0, timepoints * SOD);
    cudaMallocManaged(&u_bvec, Nbvec * 3 * SOD);
    cudaMallocManaged(&u_bval, Nbvec * SOD);
    cudaMallocManaged(&u_TD, Nbvec * SOD);
    cudaMallocManaged(&u_NewLut, prod * SOI);
    cudaMallocManaged(&u_NewIndex, newindexsize * SOI);
    cudaMallocManaged(&u_Flip, 3 * size * SOI);

    cudaMallocManaged(&u_label, n * SOI);
    cudaMallocManaged(&u_vf, SOD);
    //
    printf("Allocated Host Data\n");

    // Call Function to Set the Values for Host
    setup_data(u_dx2, u_dx4, u_SimP, u_D4Swc, u_NewLut, u_NewIndex, u_Flip, simparam, swc_trim, mdx2, mdx4,
               u_AllData, u_Reflections, u_uref, u_T2, u_T, u_SigRe, u_Sig0, u_bvec, u_bval, u_TD,
               lut, indexarr, bounds, size, iter, nrow, prod, newindexsize, sa_size, Nbvec, timepoints, NC, n, u_vf, u_label);
    // option for printing in kernel
    bool debug = false;
    double3 point = make_double3(u_D4Swc[0].x, u_D4Swc[0].y, u_D4Swc[0].z);
    // Create Random State Pointer Pointers
    curandStatePhilox4_32_10_t *deviceState;

    cudaEventRecord(start_c);
    // Allocate Memory on Device
    gpuErrchk(cudaMalloc((curandStatePhilox4_32_10_t **)&deviceState, size * sizeof(curandStatePhilox4_32_10_t)));

    // Set Values for Device
    setup_kernel<<<grid, block>>>(deviceState, 1); // initialize the random states



    int3 u_Bounds = make_int3(bounds.x, bounds.y, bounds.z);
    int3 u_IndexSize = make_int3(index_dims[0], index_dims[1], index_dims[2]);

    // Prefetch data asynchronously
    cudaMemPrefetchAsync(&u_D4Swc, nrow * SOD4, cudaCpuDeviceId);
    cudaMemPrefetchAsync(&u_dx2, 6 * iter * SOD, cudaCpuDeviceId);
    cudaMemPrefetchAsync(&u_dx4, 15 * iter * SOD, cudaCpuDeviceId);
    cudaMemPrefetchAsync(&u_SimP, 10 * SOD, cudaCpuDeviceId);
    cudaMemPrefetchAsync(&mdx2, 6 * iter * SOD, cudaCpuDeviceId);
    cudaMemPrefetchAsync(&mdx4, 15 * iter * SOD, cudaCpuDeviceId);
    cudaMemPrefetchAsync(&u_AllData, 3 * sa_size * SOD, cudaCpuDeviceId);
    cudaMemPrefetchAsync(&u_Reflections, 3 * iter * size * SOD, cudaCpuDeviceId);
    cudaMemPrefetchAsync(&u_uref, 3 * iter * size * SOD, cudaCpuDeviceId);
    cudaMemPrefetchAsync(&u_T2, Nc * SOD, cudaCpuDeviceId);
    cudaMemPrefetchAsync(&u_T, Nc * SOD, cudaCpuDeviceId);
    cudaMemPrefetchAsync(&u_SigRe, Nbvec * timepoints * SOD, cudaCpuDeviceId);
    cudaMemPrefetchAsync(&u_Sig0, timepoints * SOD, cudaCpuDeviceId);
    cudaMemPrefetchAsync(&u_bvec, Nbvec * 3 * SOD, cudaCpuDeviceId);
    cudaMemPrefetchAsync(&u_bval, Nbvec * SOD, cudaCpuDeviceId);
    cudaMemPrefetchAsync(&u_TD, Nbvec * SOD, cudaCpuDeviceId);
    cudaMemPrefetchAsync(&u_NewLut, prod * SOI, cudaCpuDeviceId);
    cudaMemPrefetchAsync(&u_NewIndex, newindexsize * SOI, cudaCpuDeviceId);
    cudaMemPrefetchAsync(&u_Flip, 3 * size * SOI, cudaCpuDeviceId);

    // new
    cudaMemPrefetchAsync(&u_label, n * SOI, cudaCpuDeviceId);
    cudaMemPrefetchAsync(&u_vf, SOD, cudaCpuDeviceId);

    // determine the volume fraction
    volfrac<<<grid, block>>>(deviceState, u_Bounds, u_D4Swc, u_NewLut, u_NewIndex, u_IndexSize, n, u_label, u_vf);
    cudaEventRecord(stop_c);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop_c);
    cudaEventElapsedTime(&milliseconds, start_c, stop_c);
    printf("Volume Fraction Kernel took %f seconds\n", milliseconds / 1e3);
    printf("Volume Fraction: %f\n", u_vf[0]);
    printf("Freeing Device Data\n");
    // Free Device Memory
    gpuErrchk(cudaFree(u_label));
    gpuErrchk(cudaFree(u_vf));

    /**
     * Call Kernel
     */
    {
        printf("Simulating...\n");
        simulate<<<grid, block>>>(u_AllData, u_dx2, u_dx4, u_Bounds, deviceState, u_SimP,
                                  u_D4Swc, u_NewLut, u_NewIndex, u_IndexSize, size,
                                  iter, debug, point, SaveAll, u_Reflections, u_uref, u_Flip,
                                  u_T2, u_T, u_Sig0, u_SigRe, u_bvec, u_bval, u_TD);
        cudaEventRecord(stop_c);
    }

    // Wait for results
    cudaDeviceSynchronize();

    /**
     * Copy Results From Device to Host
     */
    printf("Copying back to Host\n");
    cudaEventSynchronize(stop_c);
    cudaEventElapsedTime(&milliseconds, start_c, stop_c);
    clock_t end = clock();
    printf("Kernel took %f seconds\n", milliseconds / 1e3);
    printf("Writing results: ");

    auto t1 = high_resolution_clock::now();
    // // Write Results
    // {
    //     std::string outpath = root + "/results";
    //     writeResults(w_swc, u_SimP, u_dx2, mdx2, u_dx4, mdx4, u_T, u_Reflections, u_uref, u_Sig0, u_SigRe, u_AllData, iter, size, nrow, timepoints, Nbvec, sa_size, SaveAll, outpath, binaryFile, jsonFile);
    // }
    // Write New Results
    {   
        std::string outpath = root + "/results_new";
        writeResults_new(w_swc, u_SimP, u_dx2, mdx2, u_dx4, mdx4, u_T, u_Reflections, u_uref, u_Sig0, u_SigRe, u_AllData, iter, size, nrow, timepoints, Nbvec, sa_size, SaveAll, outpath, binaryFile, jsonFile);
    }

    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    printf("%f seconds\n", ms_double.count() / 1e3);

    // Free Memory
    {
        printf("Freeing Memory: ");
        gpuErrchk(cudaFree(u_dx2));
        gpuErrchk(cudaFree(u_dx4));
        gpuErrchk(cudaFree(u_SimP));
        gpuErrchk(cudaFree(u_D4Swc));
        gpuErrchk(cudaFree(u_NewLut));
        gpuErrchk(cudaFree(u_NewIndex));
        gpuErrchk(cudaFree(u_AllData));

        // Reflection Variables
        gpuErrchk(cudaFree(u_Reflections));
        gpuErrchk(cudaFree(u_uref));
        gpuErrchk(cudaFree(u_Flip));

        // Signal Variables
        gpuErrchk(cudaFree(u_T2));
        gpuErrchk(cudaFree(u_T));
        gpuErrchk(cudaFree(u_SigRe));
        gpuErrchk(cudaFree(u_Sig0));
        gpuErrchk(cudaFree(u_bvec));
        gpuErrchk(cudaFree(u_bval));
        gpuErrchk(cudaFree(u_TD));
        gpuErrchk(cudaFree(deviceState));
    }

    printf("Done!\n");
    return 0;
}
