
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

using std::cout;
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

__global__ void setup_kernel(curandStatePhilox4_32_10_t *state, unsigned long seed) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void
simulate(double *A, double *dx2, int *Bounds, int *LUT, int *IndexArray, double *SWC, bool *conditionalExample,
         curandStatePhilox4_32_10_t *state, double *SimulationParams, double4 *d4swc, int *nlut, int *NewIndex,
         int *IndexSize, int size, const int pairmax, int iter,
         bool debug) {


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
        if (gid < 3) {
            printf("double4 swc[0]->\nx: %f \t y: %f \t z: %f \t r: %f\n", d4swc[0].x, d4swc[0].y, d4swc[0].z,
                   d4swc[0].w);
            printf("double4 swc[1]->\nx: %f \t y: %f \t z: %f \t r: %f\n", d4swc[1].x, d4swc[1].y, d4swc[1].z,
                   d4swc[1].w);
            printf("particle_num: %f step_num: %f step_size: %f perm_prob: %f init_in: %f D0: %f d: %f scale: %f tstep: %f vsize: %f\n",
                   particle_num, step_num, step_size, perm_prob, init_in, D0, d, scale, tstep, vsize);
        }

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

        int2 parstate;
        bool completes;
        bool flag;

        double permprob = 0.0;
        double step = 1.0;
        double res = 0.5;

        double dist2;

        // init local state var
        curandStatePhilox4_32_10_t localstate = state[gid];

        // set particle initial position
        xi = curand_uniform4_double(&localstate);

        // todo: random initial state
        A[gx.x] = xi.x * (double) Bounds[0];
        A[gx.y] = xi.y * (double) Bounds[1];
        A[gx.z] = xi.z * (double) Bounds[2];

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
                    // printf("id: %d\n",id);
                    int id_test = s2i(floorpos, b_int3);
                    printf("X: %d\tY: %d\tZ: %d -> %d\t LUT[%d]: %d\n", floorpos.x, floorpos.y, floorpos.z, id_test,
                           id_test, nlut[id_test]);

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
                    for (int page = 0; page < i_int3.z; page++) {
                        int3 c_new = make_int3(test_lutvalue, 0, page);
                        int3 p_new = make_int3(test_lutvalue, 1, page);
                        vindex.x = NewIndex[s2i(c_new, i_int3)] - 1;
                        vindex.y = NewIndex[s2i(p_new, i_int3)] - 1;
                        printf("x: %4.1d y: %4.1d z: %4.1d\t index: %4.1d\n", c_new.x, c_new.y, c_new.z, vindex.x);
                        printf("x: %4.1d y: %4.1d z: %4.1d\t index: %4.1d\n", p_new.x, p_new.y, p_new.z, vindex.y);


                        if ((vindex.x) != -1) {
                            child.x = (double) d4swc[vindex.x].x;
                            child.y = (double) d4swc[vindex.x].y;
                            child.z = (double) d4swc[vindex.x].z;
                            child.w = (double) d4swc[vindex.x].w;
                            printf("CHILD\tx:%2.4f y:%2.4f z:%2.4f r:%2.4f\n", child.x, child.y, child.z, child.w);

                            // extract parent values
                            parent.x = (double) SWC[vindex.y].x;
                            parent.y = (double) SWC[vindex.y].y;
                            parent.z = (double) SWC[vindex.y].z;
                            parent.w = (double) SWC[vindex.y].w;
                            printf("PARENT\tx:%2.4f y:%2.4f z:%2.4f r:%2.4f\n", parent.x, parent.y, parent.z, parent.w);
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
                                page = pairmax;
                            }
                        }

                        // if the value of the index array is -1 we have checked all pairs for this particle.
                        else {
                            // end for p loop
                            p = pairmax;
                        }
                    }

//                    // warn this is unchecked
//                    int value = LUT[id];
//                    if (value >= 0) {
//                        // const int pmax_2 = pairmax+pairmax;
//                        // extract value of indexarray @ index
//
//                        for (int p = 0; p < pairmax; p++) {
//                            // extract the childId and parentId
//                            vindex.x = IndexArray[pairmax * value + 2 * p + 0];
//                            vindex.y = IndexArray[pairmax * value + 2 * p + 1];
//                            // check validity of connection given some index array values will be empty
//                            if ((vindex.x) != -1) {
//                                // extract child values
//                                child.x = (double) SWC[vindex.x * 4 + 0];
//                                child.y = (double) SWC[vindex.x * 4 + 1];
//                                child.z = (double) SWC[vindex.x * 4 + 2];
//                                child.w = (double) SWC[vindex.x * 4 + 3];
//
//                                // extract parent values
//                                parent.x = (double) SWC[vindex.y * 4 + 0];
//                                parent.y = (double) SWC[vindex.y * 4 + 1];
//                                parent.z = (double) SWC[vindex.y * 4 + 2];
//                                parent.w = (double) SWC[vindex.y * 4 + 3];
//
//                                //distance squared between child parent
//                                dist2 = ((parent.x - child.x) * (parent.x - child.x)) +
//                                        ((parent.y - child.y) * (parent.y - child.y)) +
//                                        ((parent.z - child.z) * (parent.z - child.z));
//
//                                // determine whether particle is inside this connection
//                                bool inside = swc2v(nextpos, child, parent, dist2);
//
//                                // if it is inside the connection we don't need to check the remaining.
//                                if (inside) {
//                                    // update the particles state
//                                    parstate.y = 1;
//                                    // end for p loop
//                                    p = pairmax;
//                                }
//                            }
//                                // if the value of the index array is -1 we have checked all pairs for this particle.
//                            else {
//                                // end for p loop
//                                p = pairmax;
//                            }
//                        }
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


            d2.x = (A[gx.x] - xnot.x) * res;
            d2.y = (A[gx.y] - xnot.y) * res;
            d2.z = (A[gx.z] - xnot.z) * res;

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
    std::string path = "/homes/9/bs244/Desktop/cudacodes/temp/cudaprep/data";
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


    /* 
    todo figure out why lower values of size and iteration causes bug.
    -- This appears to be the cutoff
    int size = 65;
    int iter = 65;
    */
    int size = 100;
    int iter = 100;
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
    // int boundx = (int) 10;
    // int boundy = (int) 10;
    // int boundz = (int) 10;

    int bx = (int) (boundx);
    int by = (int) (boundy);
    int bz = (int) (boundz);
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
    // 1	6	6	6	2	1
    // 2	16	6	6	2	1

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


    // stride + bx * (by * z + y) + x
    // 8,5,3 -> 698 ||| 7,4,2 -> 698
    int id1 = 0 + (boundx) * ((boundy) * 2 + 4) + 7;
    printf("lut[%d]: %d\n", id1, lut[id1]);


    std::vector <uint64_t> indexarr = sim.getIndex();

    // for (int i=0;i<indexarr.size();i++)
    // {
    //     printf("indexarr[%d],%d\n",i,indexarr[i]);
    // }

    // dx 735 
    // dy 2 
    // dz 2 


    // (x: 0,y:0,z:1);
    // stride + dx * (dy*z + y) + x;
    id1 = 0 + 735 * (2 * 1 + 0) + 0;
    printf("indexarr[%d]: %d\n", id1, indexarr[id1]);
    // (x: [1],y:[0:1],z:[0]);
    // stride + dx * (dy*z + y) + x;
    int lx;
    int ly;
    int lz;

    lx = 1;
    ly = 0;
    lz = 0;
    id1 = 0 + 735 * (2 * lz + ly) + lx;
    printf("indexarr[%d]: %d\n", id1, indexarr[id1]);
    ly = 1;
    id1 = 0 + 735 * (2 * lz + ly) + lx;
    printf("indexarr[%d]: %d\n", id1, indexarr[id1]);

    lx = 343;
    ly = 0;
    lz = 0;
    id1 = 0 + 735 * (2 * lz + ly) + lx;
    printf("indexarr[%d]: %d\n", id1, indexarr[id1]);






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
    int *hostLookup;
    bool *hostLogicalVector;
    int *hostIndexArray;
    double *hostSWC;
    int indexsize = pairmax * npair;
    double *hostdx2;
    double *hostSimP;
    int *hostNewLut;
    int *hostNewIndex;
    int *hostIndexSize;

    double4 *hostD4Swc;

    // Alloc Memory for Host Pointers
    h_a = (double *) malloc(random_bytes);
    hostBounds = (int *) malloc(3 * sizeof(int));
    hostLogicalVector = (bool *) malloc(size * sizeof(bool));
    hostLookup = (int *) malloc(prod * sizeof(int));
    hostIndexArray = (int *) malloc(pairmax * 2 * npair * sizeof(int));
    hostSWC = (double *) malloc(4 * indexsize * sizeof(double));
    hostdx2 = (double *) malloc(6 * iter * sizeof(double));
    hostSimP = (double *) malloc(10 * sizeof(double));
    hostD4Swc = (double4 *) malloc(nrow * sizeof(double4));
    hostNewLut = (int *) malloc(prod * sizeof(int));
    hostNewIndex = (int *) malloc(newindexsize * sizeof(int));
    hostIndexSize = (int *) malloc(3 * sizeof(int));


    // Set Values for Host
    memset(hostIndexArray, 0, pairmax * 2 * npair * sizeof(int));
    memset(hostSWC, 0, indexsize * sizeof(double));
    memset(hostdx2, 0, 6 * iter * sizeof(double));


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
    int *deviceBounds;
    int *deviceLookup;
    bool *deviceLogicalVector;
    int *deviceIndexArray;
    double *deviceSWC;
    double *devicedx2;
    double *deviceSimP;
    double4 *deviced4Swc;
    int *deviceNewLut;
    int *deviceNewIndex;
    int *deviceIndexSize;

    clock_t start = clock();
    // Allocate Memory on Device
    cudaMalloc((int **) &deviceLookup, prod * sizeof(int));
    cudaMalloc((int **) &deviceIndexArray, pairmax * 2 * npair * sizeof(int));
    cudaMalloc((double **) &deviceSWC, 4 * indexsize * sizeof(double));
    cudaMalloc((double **) &d_a, random_bytes);
    cudaMalloc((int **) &deviceBounds, 3 * sizeof(int));
    cudaMalloc((bool **) &deviceLogicalVector, size * sizeof(bool));
    cudaMalloc((curandStatePhilox4_32_10_t * *) & deviceState, size * sizeof(curandStatePhilox4_32_10_t));
    cudaMalloc((double **) &devicedx2, 6 * iter * sizeof(double));
    cudaMalloc((double **) &deviceSimP, 10 * sizeof(double));
    cudaMalloc((double4 * *) & deviced4Swc, nrow * sizeof(double4));
    cudaMalloc((int **) &deviceNewLut, prod * sizeof(int));
    cudaMalloc((int **) &deviceNewIndex, newindexsize * sizeof(int));
    cudaMalloc((int **) &deviceIndexSize, 3 * sizeof(int));





    // Set Values for Device
    cudaMemcpy(deviceBounds, hostBounds, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceLookup, hostLookup, prod * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceLogicalVector, hostLogicalVector, size * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceIndexArray, hostIndexArray, pairmax * 2 * npair * sizeof(int), cudaMemcpyHostToDevice);
    //todo make deviceSWC use double4 and values from disk.
    cudaMemcpy(deviceSWC, hostSWC, 4 * indexsize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devicedx2, hostdx2, 6 * iter * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceSimP, hostSimP, 10 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviced4Swc, hostD4Swc, nrow * sizeof(double4), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceNewLut, hostNewLut, prod * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceNewIndex, hostNewIndex, newindexsize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceIndexSize, hostIndexSize, 3 * sizeof(int), cudaMemcpyHostToDevice);




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
                              deviceLogicalVector, deviceState, deviceSimP, deviced4Swc, deviceNewLut, deviceNewIndex,
                              deviceIndexSize, size, pairmax, iter, debug);
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
        // printf("xx: %.4f \t xy: %.4f \t xz: %.4f\t yy: %.4f\t yz: %.4f\t zz: %.4f\t \n", hostdx2[i * 6 + 0],
        //    hostdx2[i * 6 + 1], hostdx2[i * 6 + 2], hostdx2[i * 6 + 3], hostdx2[i * 6 + 4], hostdx2[i * 6 + 5]);
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
    cudaFree(deviceSimP);
    cudaFree(deviced4Swc);



    /**
     * Free Host Data
     */
    free(h_a);
    free(hostBounds);
    free(hostLookup);
    free(hostLogicalVector);
    free(hostIndexArray);
    free(hostdx2);
    free(hostSWC);
    free(hostSimP);
    free(hostD4Swc);
    return 0;
}
