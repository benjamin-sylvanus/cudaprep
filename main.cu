#include "./src/simreader.h"
#include "./src/simulation.h"
#include "./src/particle.h"
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

using std::cout;
using std::cin;
using std::endl;

__global__ void setup_kernel(curandStatePhilox4_32_10_t *state, unsigned long seed) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void simulate(double *savedata, double *dx2, int *Bounds, curandStatePhilox4_32_10_t *state, double *SimulationParams,
         double4 *d4swc, int *nlut, int *NewIndex, int *IndexSize, int size, int iter, bool debug) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;

    if (gid < size) {
        double step_size = SimulationParams[2];
        double perm_prob = SimulationParams[3];
        double init_in = SimulationParams[4];
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
        double permprob = 0.0;
        double step = step_size;

        // init local state var
        curandStatePhilox4_32_10_t localstate = state[gid];
        xi = curand_uniform4_double(&localstate);

        // initialize position inside cell
        A = initPosition(gid, dx2, Bounds, state, SimulationParams, d4swc, nlut, NewIndex, IndexSize,
                         size, iter, debug);

        // record initial position
        xnot = make_double3(A.x, A.y, A.z);

        // flag is initially false
        flag = 0;

        // state is based on intialization conditions if particles are required to start inside then parstate -> [1,1]
        parstate = make_int2(1, 1);

        // iterate over steps
        for (int i = 0; i < iter; i++) {

            if (flag == 0) {
                // generate uniform randoms for step
                xi = curand_uniform4_double(&localstate);

                // set next position
                setNextPos(nextpos, A, xi, step);

                // floor of next position -> check voxels
                floorpos = make_int3((int) nextpos.x, (int) nextpos.y, (int) nextpos.z);

                // upper bounds of lookup table
                upper = make_int3(floorpos.x < b_int3.x, floorpos.y < b_int3.y, floorpos.x < b_int3.z);

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

                    // for each connnction check if particle inside
                    for (int page = 0; page < i_int3.z; page++) {

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
                            dist2 = pow(parent.x - child.x, 2) + pow(parent.y - child.y, 2) +
                                    pow(parent.z - child.z, 2);

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
                            parstate.y = 1;
                        }
                    }
                }

                // determine if step executes
                completes = xi.w < permprob;

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
                }
            } else {
                // update flag for next step
                flag = false;
            }
            /**
             * Store Results Function
             */
            diffusionTensor(A, xnot, vsize, dx2, savedata, d2, i);
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

    setupSimulation();
    system("clear");

    cout << "---Welcome---\n";

    cout << "1. Standard Simulation \n";
    cout << "2. Change Configuration \n";
    cout << "3. Custom Configuration \n";
    cout << "4. Help \n"
    cout << "Command List: \n";

    int size = 10;
    int iter = 10;

    /**
     * Read Simulation and Initialize Object
     */
    std::string path = "./data";
    simreader reader(&path);
    simulation sim(reader);
    double simparam[10];
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
        printf("%f\t", simulationparams[i]);
    }
    printf("\n");

    cout << "Enter Number of Particles: \n";
    cin >> size;
    cout << "Enter Number of Steps" << '\n';
    cin >> iter;
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
    int *hostBounds;
    double *hostdx2;
    double *hostSimP;
    int *hostNewLut;
    int *hostNewIndex;
    int *hostIndexSize;
    double4 *hostD4Swc;
    double *hostAllData;

    // Alloc Memory for Host Pointers
    hostBounds = (int *) malloc(3 * sizeof(int));
    hostdx2 = (double *) malloc(6 * iter * sizeof(double));
    hostSimP = (double *) malloc(10 * sizeof(double));
    hostD4Swc = (double4 *) malloc(nrow * sizeof(double4));
    hostNewLut = (int *) malloc(prod * sizeof(int));
    hostNewIndex = (int *) malloc(newindexsize * sizeof(int));
    hostIndexSize = (int *) malloc(3 * sizeof(int));
    hostAllData = (double *) malloc(3 * iter * size * sizeof(double));

    // Set Values for Host
    memset(hostdx2, 0.0, 6 * iter * sizeof(double));
    memset(hostAllData, 0.0, 3 * iter * size * sizeof(double));

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
    double *devicedx2;
    int *deviceBounds;
    double *deviceSimP;
    double4 *deviced4Swc;
    int *deviceNewLut;
    int *deviceNewIndex;
    int *deviceIndexSize;
    double *deviceAllData;

    clock_t start = clock();
    // Allocate Memory on Device
    cudaMalloc((double **) &devicedx2, 6 * iter * sizeof(double));
    cudaMalloc((int **) &deviceBounds, 3 * sizeof(int));
    cudaMalloc((curandStatePhilox4_32_10_t * *) & deviceState, size * sizeof(curandStatePhilox4_32_10_t));
    cudaMalloc((double **) &deviceSimP, 10 * sizeof(double));
    cudaMalloc((double4 * *) & deviced4Swc, nrow * sizeof(double4));
    cudaMalloc((int **) &deviceNewLut, prod * sizeof(int));
    cudaMalloc((int **) &deviceNewIndex, newindexsize * sizeof(int));
    cudaMalloc((int **) &deviceIndexSize, 3 * sizeof(int));
    cudaMalloc((double **) &deviceAllData, 3 * iter * size * sizeof(double));

    // Set Values for Device
    cudaMemcpy(devicedx2, hostdx2, 6 * iter * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceBounds, hostBounds, 3 * sizeof(int), cudaMemcpyHostToDevice);
    setup_kernel<<<grid, block>>>(deviceState, 1);
    cudaMemcpy(deviceSimP, hostSimP, 10 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviced4Swc, hostD4Swc, nrow * sizeof(double4), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceNewLut, hostNewLut, prod * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceNewIndex, hostNewIndex, newindexsize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceIndexSize, hostIndexSize, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceAllData, hostAllData, 3 * iter * size * sizeof(double), cudaMemcpyHostToDevice);

    /**
     * Initalize Random Stream
     */
    // option for printing in kernel
    bool debug = false;

    /**
     * Call Kernel
    */
    simulate<<<grid, block>>>(deviceAllData, devicedx2, deviceBounds, deviceState, deviceSimP, deviced4Swc,
                              deviceNewLut, deviceNewIndex,
                              deviceIndexSize, size, iter, debug);
    // Wait for results
    cudaDeviceSynchronize();
    clock_t end = clock();
    double gpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("kernel took %f seconds\n", gpu_time_used);

    /**
     * Copy Results From Device to Host
     */
    cudaMemcpy(hostdx2, devicedx2, 6 * iter * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostAllData, deviceAllData, 3 * iter * size * sizeof(double), cudaMemcpyDeviceToHost);

    end = clock();
    gpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("kernel took %f seconds\n", gpu_time_used);

    /**
     * Free Device Data
     */
    cudaFree(deviceBounds);
    cudaFree(deviceState);
    cudaFree(devicedx2);
    cudaFree(deviceSimP);
    cudaFree(deviced4Swc);
    cudaFree(deviceNewIndex);
    cudaFree(deviceIndexSize);
    cudaFree(deviceAllData);

    /**
    * Write Results to file
    We need to write
    hostdx2
    */
    double t[iter];
    double tstep = hostSimP[8];
    double mdx_2[6 * iter];
    for (int i = 0; i < iter; i++) {
        t[i] = tstep * i;
        mdx_2[6 * i + 0] = (hostdx2[6 * i + 0] / size) / (2.0 * t[i]);
        mdx_2[6 * i + 1] = (hostdx2[6 * i + 1] / size) / (2.0 * t[i]);
        mdx_2[6 * i + 2] = (hostdx2[6 * i + 2] / size) / (2.0 * t[i]);
        mdx_2[6 * i + 3] = (hostdx2[6 * i + 3] / size) / (2.0 * t[i]);
        mdx_2[6 * i + 4] = (hostdx2[6 * i + 4] / size) / (2.0 * t[i]);
        mdx_2[6 * i + 5] = (hostdx2[6 * i + 5] / size) / (2.0 * t[i]);
    }

    std::ofstream fdx2out("./results/dx2.txt");
    std::ofstream fmdx2out("./results/mdx2.txt");
    fdx2out.precision(15);
    fmdx2out.precision(15);

    for (int i = 0; i < iter; i++) {

        for (int j = 0; j < 6; j++) {

            if (j == 5) {
                fdx2out << hostdx2[i * 6 + j] << endl;
            } else {
                fdx2out << hostdx2[i * 6 + j] << "\t";
            }
            if (j == 5) {
                fmdx2out << mdx_2[i * 6 + j] << endl;
            } else {
                fmdx2out << mdx_2[i * 6 + j] << "\t";
            }
        }
    }

    fdx2out.close();
    fmdx2out.close();
    int3 dataindex = make_int3(iter, size, 3);
    int3 dataid;
    for (int j = 0; j < size; j++) {
        for (int i = 0; i < iter; i++) {
            dataid.x = dataindex.x * (dataindex.y * 0 + i) + j;
            dataid.y = dataindex.x * (dataindex.y * 1 + i) + j;
            dataid.z = dataindex.x * (dataindex.y * 2 + i) + j;
            // printf("particle(id: %d,step: %d): %f %f %f\n",j,i,hostAllData[dataid.x],hostAllData[dataid.y],hostAllData[dataid.z]);
        }
    }

    /**
     * Free Host Data
     */
    free(hostBounds);
    free(hostdx2);
    free(hostSimP);
    free(hostD4Swc);
    free(hostNewIndex);
    free(hostIndexSize);
    free(hostAllData);
    return 0;
}
