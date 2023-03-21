#include "./src/simreader.h"
#include "./src/simulation.h"
#include "./src/controller.h"
#include "./src/viewer.h"
#include "./src/funcs.h"
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

using std::cout;
using std::cin;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

__global__ void setup_kernel(curandStatePhilox4_32_10_t *state, unsigned long seed) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}
// __global__ void simulate(double * savedata,double *dx2, int *Bounds, curandStatePhilox4_32_10_t *state, double *SimulationParams,
//          double4 *d4swc, int *nlut, int *NewIndex, int *IndexSize, int size, int iter, bool debug)
__global__ void simulate(double *dx2, double *dx4, int *Bounds, curandStatePhilox4_32_10_t *state, double *SimulationParams,
         double4 *d4swc, int *nlut, int *NewIndex, int *IndexSize, int size, int iter, bool debug, double3 point) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;

    if (gid < size) {
        double step_size = SimulationParams[2];
        double perm_prob = SimulationParams[3];
        int init_in = (int) SimulationParams[4];
        double vsize = SimulationParams[9];
        double3 A;
        int2 parstate;
        int3 gx = make_int3(3 * gid + 0, 3 * gid + 1, 3 * gid + 2);

        // define variables for loop
        double4 xi;
        double3 nextpos;
        double3 xnot;
        int3 upper;
        int3 lower;
        int3 floorpos;
        int3 b_int3 = make_int3(Bounds[0], Bounds[1], Bounds[2]);
        int3 i_int3 = make_int3(IndexSize[0], IndexSize[1], IndexSize[2]);
        double3 d2 = make_double3(0.0, 0.0, 0.0);
        bool completes;
        bool flag;
        double step = step_size;

        // init local state var
        curandStatePhilox4_32_10_t localstate = state[gid];
        xi = curand_uniform4_double(&localstate);

        // initialize position inside cell
        A = initPosition(gid, dx2, Bounds, state, SimulationParams, d4swc, nlut, NewIndex, IndexSize,
                         size, iter,init_in, debug, point);

        // record initial position
        xnot = make_double3(A.x, A.y, A.z);

        // flag is initially false
        flag = 0;

        // state is based on intialization conditions if particles are required to start inside then parstate -> [1,1]
        // todo figure out how to get parstate from init position function... requires a global parstate.
        parstate = make_int2(0, 1);

        // iterate over steps
        for (int i = 0; i < iter; i++) {

          // printf("i: %d\tgid: %d\n",i,gid);

            if (flag == 0) {
                // generate uniform randoms for step
                xi = curand_uniform4_double(&localstate);

                // set next position
                nextpos.x = A.x + ((2.0 * xi.x - 1.0) * step);
                nextpos.y = A.y + ((2.0 * xi.y - 1.0) * step);
                nextpos.z = A.z + ((2.0 * xi.z - 1.0) * step);

                // floor of next position -> check voxels
                floorpos = make_int3((int) nextpos.x, (int) nextpos.y, (int) nextpos.z);

                // upper bounds of lookup table
                upper = make_int3(floorpos.x < b_int3.x, floorpos.y < b_int3.y, floorpos.z < b_int3.z);

                // lower bounds of lookup table
                lower = make_int3(floorpos.x >= 0, floorpos.y >= 0, floorpos.z >= 0);

                // position inside the bounds of volume -> state of next position true : false
                parstate.y = (lower.x && lower.y && lower.z && upper.x && upper.y && upper.z) ? 1 : 0;
                // extract value of lookup @ index
                if (parstate.y) {
                    // reset particle state for next conditionals
                    parstate.y = 0;

                    // scuffed sub2ind function
                    int id_test = s2i(floorpos, b_int3);

                    // extract lookup table value
                    int test_lutvalue = nlut[id_test];

                    // child parent indicies
                    int2 vindex;

                    // parent swc values
                    double4 parent;

                    // child swc values
                    double4 child;

                    // distance^2 from child to parent
                    double dist2;

                    // for each connection check if particle inside
                    for (int page = 0; page < i_int3.z; page++) {
                        // printf("I: %d\tGID: %d\tPage: %d\n", i,gid,page);

                        // create a subscript indices
                        int3 c_new = make_int3(test_lutvalue, 0, page);
                        int3 p_new = make_int3(test_lutvalue, 1, page);

                        // convert subscripted index to linear index and get value from Index Array
                        vindex.x = NewIndex[s2i(c_new, i_int3)] - 1;
                        vindex.y = NewIndex[s2i(p_new, i_int3)] - 1;

                        if ((vindex.x) != -1) {
                            //extract child parent values from swc
                            child = d4swc[vindex.x];
                            parent = d4swc[vindex.y];

                            // calculate euclidean distance
                            dist2 = pow(parent.x - child.x, 2) + pow(parent.y - child.y, 2) + pow(parent.z - child.z, 2);

                            // determine whether particle is inside this connection
                            bool inside = swc2v(nextpos, child, parent, dist2);

                            // if it is inside the connection we don't need to check the remaining.
                            if (inside) {
                                // update the particles state
                                parstate.y = 1;

                                // end for p loop
                                page = i_int3.z;
                            }
                        }

                            // if the value of the index array is -1 we have checked all pairs for this particle.
                        else {
                            // end for p loop
                            page = i_int3.z;
                            parstate.y = 0;
                        }
                    }
                }

                // determine if step executes
                completes = xi.w < perm_prob;

                // if the next position is inside we update the position
                if (parstate.y) {
                    A = nextpos;
                } else {
                    // if the particle exits successfully update the position
                    if (completes && parstate.x) {
                        A = nextpos;
                    }
                    // if the particle fails to exit set flag to true
                    if (!completes && parstate.x && !parstate.y) {
                        flag = true;
                    }
                    if (!parstate.x && !parstate.y)
                    {
                        A = nextpos;
                    }
                }

            } else {
                // update flag for next step
                flag = false;
            }
            /**
             * Store Results Function
             */
            // diffusionTensor(A, xnot, vsize, dx2, savedata, d2, i, gid, iter, size);
            d2.x = fabs((A.x - xnot.x) * vsize);
            d2.y = fabs((A.y - xnot.y) * vsize);
            d2.z = fabs((A.z - xnot.z) * vsize);

            // Diffusion Tensor
            atomicAdd(&dx2[6 * i + 0], d2.x * d2.x);
            atomicAdd(&dx2[6 * i + 1], d2.x * d2.y);
            atomicAdd(&dx2[6 * i + 2], d2.x * d2.z);
            atomicAdd(&dx2[6 * i + 3], d2.y * d2.y);
            atomicAdd(&dx2[6 * i + 4], d2.y * d2.z);
            atomicAdd(&dx2[6 * i + 5], d2.z * d2.z);

            // Kurtosis Tensor
            atomicAdd(&dx4[15* i +0],d2.x * d2.x * d2.x * d2.x);
            atomicAdd(&dx4[15* i +1],d2.x * d2.x * d2.x * d2.y);
            atomicAdd(&dx4[15* i +2],d2.x * d2.x * d2.x * d2.z);
            atomicAdd(&dx4[15* i +3],d2.x * d2.x * d2.y * d2.y);
            atomicAdd(&dx4[15* i +4],d2.x * d2.x * d2.y * d2.z);
            atomicAdd(&dx4[15* i +5],d2.x * d2.x * d2.z * d2.z);
            atomicAdd(&dx4[15* i +6],d2.x * d2.y * d2.y * d2.y);
            atomicAdd(&dx4[15* i +7],d2.x * d2.y * d2.y * d2.z);
            atomicAdd(&dx4[15* i +8],d2.x * d2.y * d2.z * d2.z);
            atomicAdd(&dx4[15* i +9],d2.x * d2.z * d2.z * d2.z);
            atomicAdd(&dx4[15* i +10],d2. y *d2. y *d2. y *d2.y);
            atomicAdd(&dx4[15* i +11],d2. y *d2. y *d2. y *d2.z);
            atomicAdd(&dx4[15* i +12],d2. y *d2. y *d2. z *d2.z);
            atomicAdd(&dx4[15* i +13],d2. y *d2. z *d2. z *d2.z);
            atomicAdd(&dx4[15* i +14],d2. z *d2. z *d2. z *d2.z);

            // int3 dix = make_int3(size, iter,3);

            // int3 did[4];
            // did[0] = make_int3(gid,i,0);
            // did[1] = make_int3(gid,i,1);
            // did[2] = make_int3(gid,i,2);
            // did[3] = make_int3(s2i(did[0],dix),s2i(did[1],dix),s2i(did[2],dix));

            // savedata[did[3].x] = A.x;
            // savedata[did[3].y] = A.y;
            // savedata[did[3].z] = A.z;
        }
    }
}

void setupSimulation() {
    std::string path = "./data";
    simreader reader(&path);
    simulation sim(reader);
    sim.setStep_num(100);
    sim.setParticle_num(100);
}


int main() {
    cudaEvent_t start_c, stop_c;
    cudaEventCreate(&start_c);
    cudaEventCreate(&stop_c);
    float milliseconds = 0;
    system("clear");
    int size = 10;
    int iter = 10;

    /**
     * Read Simulation and Initialize Object
     */
    std::string path = "/autofs/space/symphony_002/users/BenSylvanus/cuda/Sims/";
    std::string dataPath = path;
    dataPath.append("data");

    size_t len = path.length();
    controller control(dataPath);
    control.start();
    system("clear");

    double simparam[10];
    simulation sim = control.getSim();
    size = sim.getParticle_num();
    iter = sim.getStep_num();
    /**
     <li> particle_num = SimulationParams[0] </li>
     <li> step_num = SimulationParams[1] </li>
     <li> step_size = SimulationParams[2] </li>
     <li> perm_prob = SimulationParams[3] </li>
     <li> init_in = SimulationParams[4] </li>
     <li> D0 = SimulationParams[5] </li>
     <li> d = SimulationParams[6] </li>
     <li> scale = SimulationParams[7] </li>
     <li> tstep = SimulationParams[8] </li>
     <li> vsize = SimulationParams[9] </li>
     */
    std::vector<double> simulationparams = sim.getParameterdata();
    for (int i = 0; i < 10; i++) {
        double value = simulationparams[i];
        simparam[i] = value;
    }

    int block_size = 128;
    dim3 block(block_size);
    dim3 grid((size / block.x) + 1);

    // todo: figure out why different bound size gives bug
    std::vector <uint64_t> bounds = sim.getbounds();
    int boundx = (int) bounds[0];
    int boundy = (int) bounds[1];
    int boundz = (int) bounds[2];
    int prod = (int) (boundx * boundy * boundz);
    std::vector<double> r_swc = sim.getSwc();
    int nrow = r_swc.size() / 6;
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
    //we only need the x y z r    float milliseconds = 0; of our swc array.
    double4 swc_trim[nrow];
    double w_swc[nrow*4];
    for (int i = 0; i < nrow; i++) {
        swc_trim[i].x = r_swc[i + nrow * 1];
        swc_trim[i].y = r_swc[i + nrow * 2];
        swc_trim[i].z = r_swc[i + nrow * 3];
        swc_trim[i].w = r_swc[i + nrow * 4];
    }
    for (int i=0; i<nrow; i++)
    {
      w_swc[4*i + 0] = r_swc[i + nrow*1];
      w_swc[4*i + 1] = r_swc[i + nrow*2];
      w_swc[4*i + 2] = r_swc[i + nrow*3];
      w_swc[4*i + 3] = r_swc[i + nrow*4];
    }
    std::vector <uint64_t> lut = sim.getLut();

    // stride + bx * (by * y + z) + x
    // int id0 = 0 + (boundx) * ((boundy) * 2 + 2) + 3;
    // printf("lut[%d]: %d\n", id0, lut[id0]);
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
    // printf("%d \t %d \t %d\n", index_dims[0], index_dims[1], index_dims[2]);
    int newindexsize = index_dims[0] * index_dims[1] * index_dims[2];

    /**
     * Host Section:
     * - Create Pointers
     * - Allocate Memory
     * - Set Values
     */
    // Create Host Pointers
    printf("Creating Host Data\n");
    int *hostBounds;
    double *hostdx2;
    double *hostdx4;
    double *hostSimP;
    int *hostNewLut;
    int *hostNewIndex;
    int *hostIndexSize;
    double4 *hostD4Swc;
    // double *hostAllData;

    // Alloc Memory for Host Pointers
    hostBounds = (int *) malloc(3 * sizeof(int));
    hostdx2 = (double *) malloc(6 * iter * sizeof(double));
    hostdx4 = (double *) malloc(15 * iter * sizeof(double));
    hostSimP = (double *) malloc(10 * sizeof(double));
    hostD4Swc = (double4 *) malloc(nrow * sizeof(double4));
    hostNewLut = (int *) malloc(prod * sizeof(int));
    hostNewIndex = (int *) malloc(newindexsize * sizeof(int));
    hostIndexSize = (int *) malloc(3 * sizeof(int));
    // hostAllData = (double *) malloc(3 * iter * size * sizeof(double));

    // Set Values for Host
    memset(hostdx2, 0.0, 6 * iter * sizeof(double));
    memset(hostdx4, 0.0, 15 * iter * sizeof(double));
    // memset(hostAllData, 0.0, 3 * iter * size * sizeof(double));


    // for (int i = 0; i<3*iter*size; i++)
    //
    // {
      // hostAllData[i] = (double) i;
    // }

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
    printf("Creating Device Data.\n");
    curandStatePhilox4_32_10_t *deviceState;
    double *devicedx2;
    double *devicedx4;
    int *deviceBounds;
    double *deviceSimP;
    double4 *deviced4Swc;
    int *deviceNewLut;
    int *deviceNewIndex;
    int *deviceIndexSize;
    // double *deviceAllData;

    clock_t start = clock();
    cudaEventRecord(start_c);
    // Allocate Memory on Device
    cudaMalloc((double **) &devicedx2, 6 * iter * sizeof(double));
    cudaMalloc((double **) &devicedx4, 15 * iter * sizeof(double));
    cudaMalloc((int **) &deviceBounds, 3 * sizeof(int));
    cudaMalloc((curandStatePhilox4_32_10_t * *) & deviceState, size * sizeof(curandStatePhilox4_32_10_t));
    cudaMalloc((double **) &deviceSimP, 10 * sizeof(double));
    cudaMalloc((double4 * *) &deviced4Swc, nrow * sizeof(double4));
    cudaMalloc((int **) &deviceNewLut, prod * sizeof(int));
    cudaMalloc((int **) &deviceNewIndex, newindexsize * sizeof(int));
    cudaMalloc((int **) &deviceIndexSize, 3 * sizeof(int));
    // cudaMalloc((double **) &deviceAllData, 3 * iter * size * sizeof(double));

    // Set Values for Device
    printf("Copying Host data to Device\n");
    cudaMemcpy(devicedx2, hostdx2, 6 * iter * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devicedx4, hostdx4, 15 * iter * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceBounds, hostBounds, 3 * sizeof(int), cudaMemcpyHostToDevice);
    setup_kernel<<<grid, block>>>(deviceState, 1);
    cudaMemcpy(deviceSimP, hostSimP, 10 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviced4Swc, hostD4Swc, nrow * sizeof(double4), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceNewLut, hostNewLut, prod * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceNewIndex, hostNewIndex, newindexsize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceIndexSize, hostIndexSize, 3 * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(deviceAllData, hostAllData, 3 * iter * size * sizeof(double), cudaMemcpyHostToDevice);

    /**
     * Initalize Random Stream
     */
    // option for printing in kernel
    bool debug = false;
    double3 point = make_double3(6.0,6.0,6.0);
    /**
     * Call Kernel
    */
    printf("Simulating...\n");
    // simulate<<<grid, block>>>(deviceAllData,devicedx2, deviceBounds, deviceState, deviceSimP, deviced4Swc,
    //                           deviceNewLut, deviceNewIndex,
    //                           deviceIndexSize, size, iter, debug);
    simulate<<<grid, block>>>(devicedx2, devicedx4, deviceBounds, deviceState, deviceSimP, deviced4Swc,
                              deviceNewLut, deviceNewIndex,
                              deviceIndexSize, size, iter, debug,point);
    cudaEventRecord(stop_c);
    // Wait for results
    cudaDeviceSynchronize();

    clock_t end = clock();
    double gpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Simulation took %f seconds\n", gpu_time_used);

    /**
     * Copy Results From Device to Host
     */
    printf("Copying back to Host\n");
    cudaMemcpy(hostdx2, devicedx2, 6 * iter * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostdx4, devicedx4, 15 * iter * sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(hostAllData, deviceAllData, 3 * iter * size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop_c);
    cudaEventElapsedTime(&milliseconds, start_c, stop_c);
    end = clock();
    printf("kernel took %f seconds\n", milliseconds/1e3);
    /**
     * Free Device Data
     */
    auto t1 = high_resolution_clock::now();
    printf("Freeing Device Data: ");
    cudaFree(deviceBounds);
    cudaFree(deviceState);
    cudaFree(devicedx2);
    cudaFree(devicedx4);
    cudaFree(deviceSimP);
    cudaFree(deviced4Swc);
    cudaFree(deviceNewIndex);
    cudaFree(deviceIndexSize);
    // cudaFree(deviceAllData);
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    printf("%f seconds\n", ms_double.count()/1e3);
    printf("Writing results: ");

    // Check outdir exists
    // isdir()?  mkdir : ...
    std::string outpath = path.append("results/");
    t1 = high_resolution_clock::now();
    writeResults(hostdx2, hostdx4, hostSimP,  w_swc,  iter, size, outpath);
    t2 = high_resolution_clock::now();
    ms_double = t2 - t1;
    printf("%f seconds\n", ms_double.count()/1e3);

    /**
     * Free Host Data
     */
    free(hostBounds);
    free(hostdx2);
    free(hostdx4);
    free(hostSimP);
    free(hostD4Swc);
    free(hostNewIndex);
    free(hostIndexSize);
    // free(hostAllData);
    return 0;
}
