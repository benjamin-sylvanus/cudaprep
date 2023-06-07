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
#include <cublas_v2.h>

#define CUDART_PI_F 3.141592654f
#define PI 3.14159265358979323846
#define LDPI 3.141592653589793238462643383279502884L
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

__global__ void setup_kernel(curandStatePhilox4_32_10_t *state, unsigned long seed) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

/**
 * @brief Checks if the particle is inside the connections listed for voxel
 * @param i_int3 size of the index array
 * @param test_lutvalue index of the voxel
 * @param nextpos position of the particle
 * @param NewIndex index array
 * @param d4swc swc array
 * @return true if particle is inside the connection
 */
__device__ bool checkConnections(int3 i_int3, int test_lutvalue, double3 nextpos, int *NewIndex, double4 *d4swc) {
    int3 vindex;
    double4 child, parent;
    double dist2;

    // for each connection check if particle inside
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

            dist2 = distance(parent, child);
            printf("dist via new method: %f\n", dist2);
            dist2 = pow(parent.x - child.x, 2) + pow(parent.y - child.y, 2) + pow(parent.z - child.z, 2);
            printf("dist via old method: %f\n", dist2);

            // determine whether particle is inside this connection
            bool inside = swc2v(nextpos, child, parent, dist2);

            // if it is inside the connection we don't need to check the remaining.
            if (inside) {
                // end for p loop
                return true;
            }
        }
            // if the value of the index array is -1 we have checked all pairs for this particle.
            // checkme: how often does this happen?
        else {
            // end for p loop
            return false;
        }
    }
    return false;
}
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
__device__ void validCoord(double3 &nextpos, double3 &pos, int3 &b_int3, int3 &upper, int3 &lower, int3 &floorpos, double * reflections, double * uref, int gid, int i, int size, int iter, int * flips)
{
    double3 High = make_double3((double)b_int3.x, (double)b_int3.y, (double) b_int3.z);
    double3 Low = make_double3(0.0, 0.0, 0.0);

    // determine the index of the reflection storage should match the save data index
    int3 dix = make_int3(size, iter, 3);
    int3 did[4];
    did[0] = make_int3(gid, i, 0);
    did[1] = make_int3(gid, i, 1);
    did[2] = make_int3(gid, i, 2);
    did[3] = make_int3(s2i(did[0], dix), s2i(did[1], dix), s2i(did[2], dix));

    int fidx; // flip index for reflection

    int count = 0;
    while(true)
    {
        int3 UPPER = make_int3(nextpos.x > High.x, nextpos.y > High.y, nextpos.z > High.z);
        int3 LOWER = make_int3(nextpos.x < Low.x, nextpos.y < Low.y, nextpos.z < Low.z);

        // normal vector
        double3 normal;

        // point on plane
        double3 pointOnPlane;

        if (LOWER.x)
        {
            fidx = 6*gid + 0;
            pointOnPlane = make_double3(Low.x, nextpos.y, nextpos.z);
            normal = make_double3(1.0, 0.0, 0.0);
        }
        else if (UPPER.x)
        {
            fidx = 6*gid + 1;
            pointOnPlane = make_double3(High.x, nextpos.y, nextpos.z);
            normal = make_double3(-1.0, 0.0, 0.0);
        }
        else if (LOWER.y)
        {
            fidx = 6*gid + 2;
            pointOnPlane = make_double3(nextpos.x, Low.y, nextpos.z);
            normal = make_double3(0.0, 1.0, 0.0);
        }
        else if (UPPER.y)
        {
            fidx = 6*gid + 3;
            pointOnPlane = make_double3(nextpos.x, High.y, nextpos.z);
            normal = make_double3(0.0, -1.0, 0.0);
        }
        else if (LOWER.z)
        {
            fidx = 6*gid + 4;
            pointOnPlane = make_double3(nextpos.x, nextpos.y, Low.z);
            normal = make_double3(0.0, 0.0, 1.0);
        }
        else if (UPPER.z)
        {
            fidx = 6*gid + 5;
            pointOnPlane = make_double3(nextpos.x, nextpos.y, High.z);
            normal = make_double3(0.0, 0.0, -1.0);
        }
        else
        {
            return; // no reflection needed
        }

        // Calculate D  (Ax + By + Cz + D = 0)
        double D = -(dot(normal, pointOnPlane));

        double3 intersectionPoint;
        double3 d = pos - nextpos;

        double t1 = -((dot(normal, nextpos) + D)) / dot(normal, d);
        intersectionPoint = nextpos + d * t1;

        double3 reflectionVector = nextpos - intersectionPoint;
        reflectionVector = reflectionVector - normal * (2 * dot(reflectionVector,normal));

        // record the unreflected position
        double3 unreflected = nextpos;
        double3 intersection = intersectionPoint;
        nextpos = intersectionPoint + reflectionVector;

        printf("NextPos: %f %f %f -> %f %f %f\n", nextpos.x, nextpos.y, nextpos.z, intersectionPoint.x+reflectionVector.x, intersectionPoint.y + reflectionVector.y, intersectionPoint.z + reflectionVector.z);
        printf("Count: %d\n", count);
        count += 1;

        // store the intersection point
        reflections[did[3].x] = intersectionPoint.x;
        reflections[did[3].y] = intersectionPoint.y;
        reflections[did[3].z] = intersectionPoint.z;

        // store the unreflected vector
        uref[did[3].x] = unreflected.x;
        uref[did[3].y] = unreflected.y;
        uref[did[3].z] = unreflected.z;

        // Update the particle's position
        nextpos = intersectionPoint + reflectionVector;

        // flip the particle's direction
        flips[fidx] += 1; // no need for atomicAdd since gid is what is parallelized
    }
}



__global__ void simulate(double *savedata, double *dx2, double *dx4, int *Bounds, curandStatePhilox4_32_10_t *state,
                         double *SimulationParams,
                         double4 *d4swc, int *nlut, int *NewIndex, int *IndexSize, int size, int iter, bool debug,
                         double3 point, int SaveAll, double * Reflections, double * Uref, int * flip,
                         double * T2, double * T, double * Sig0, double * SigRe, double* BVec, double * BVal, double * TD) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;

    if (gid < size) {
        double step_size = SimulationParams[2];
        double perm_prob = SimulationParams[3];
        int init_in = (int) SimulationParams[4];
        // init_in = 3;
        double vsize = SimulationParams[9];
        double3 A;
        int2 parstate;
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

        /////////////////////
        // signal variables
        {
            /*
             * double s0 = 0; // Signal weighted by T2 relaxation
             * double t[Nc_max] = {0}; // The time staying in compartments
            */
        }
        ////////////////////

        // init local state var
        curandStatePhilox4_32_10_t localstate = state[gid];
        xi = curand_uniform4_double(&localstate);

        // initialize position inside cell
        A = initPosition(gid, dx2, Bounds, state, SimulationParams, d4swc, nlut, NewIndex, IndexSize,
                         size, iter, init_in, debug, point);
                         size, iter, init_in, debug, point);

        // record initial position
        xnot = make_double3(A.x, A.y, A.z);

        // flag is initially false
        flag = false;
        flag = false;

        // state is based on intialization conditions if particles are required to start inside then parstate -> [1,1]
        // todo figure out how to get parstate from init position function... requires a global parstate.
        parstate = make_int2(1, 1);

        // parlut defines whether particle is within bounds of LUT
        int parlut = 1;
        parstate = make_int2(1, 1);

        // parlut defines whether particle is within bounds of LUT
        int parlut = 1;

        // iterate over steps
        for (int i = 0; i < iter; i++) {

            if (flag == false) {
                // generate uniform randoms for step
                xi = curand_uniform4_double(&localstate);

                // set next position
                double theta = 2 * PI * xi.x;
                double v = xi.y;
                double cos_phi = 2 * v - 1;
                double sin_phi = sqrt(1 - pow(cos_phi, 2));
                nextpos.x = A.x + (step * sin_phi * cos(theta));
                nextpos.y = A.y + (step * sin_phi * sin(theta));
                nextpos.z = A.z + (step * cos_phi);
                double theta = 2 * PI * xi.x;
                double v = xi.y;
                double cos_phi = 2 * v - 1;
                double sin_phi = sqrt(1 - pow(cos_phi, 2));
                nextpos.x = A.x + (step * sin_phi * cos(theta));
                nextpos.y = A.y + (step * sin_phi * sin(theta));
                nextpos.z = A.z + (step * cos_phi);

                // check coordinate validity
                validCoord(nextpos, A, b_int3, upper, lower, floorpos, Reflections, Uref, gid, i, size, iter, flip);

                // floor of next position -> check voxels
                floorpos = make_int3((int) nextpos.x, (int) nextpos.y, (int) nextpos.z);

                // reset particle state for next conditionals
                parstate.y = 0; // checkme: is this necessary or valid?

                // sub2ind
                int id_test = s2i(floorpos, b_int3);
                // sub2ind
                int id_test = s2i(floorpos, b_int3);

                // extract lookup table value
                int test_lutvalue = nlut[id_test];
                // extract lookup table value
                int test_lutvalue = nlut[id_test];

                // child parent indicies
                int2 vindex;
                // child parent indicies
                int2 vindex;

                // parent swc values
                double4 parent;
                // parent swc values
                double4 parent;

                // child swc values
                double4 child;
                // child swc values
                double4 child;

                // distance^2 from child to parent
                double dist2;
                // distance^2 from child to parent
                double dist2;

                // for each connection check if particle inside
                bool inside = checkConnections(i_int3, test_lutvalue, nextpos, NewIndex, d4swc);
                if (inside) {
                    // update the particles state
                    parstate.y = 1;
                } else {
                    parstate.y = 0;
                }


                // determine if step executes
                completes = xi.w < perm_prob;

                /**
                * @cases particle inside? 0 0 - update
                * @cases particle inside? 0 1 - check if updates
                * @cases particle inside? 1 0 - check if updates
                * @cases particle inside? 1 1 - update
                */

                ////////////////////////////////////////////////////////////////
                // ========================================================== //
                // ========================================================== //
                ////////////////////////////////////////////////////////////////
                ///////////////// UPDATE PARTICLE COMPARTMENT //////////////////
                ////////////////////////////////////////////////////////////////
                // ========================================================== //
                // ========================================================== //
                ////////////////////////////////////////////////////////////////

                // particle inside: [0 0] || [1 1]
                if (parstate.x == parstate.y) { A = nextpos; }

                // particle inside: [1 0]
                if (parstate.x && !parstate.y) {
                    if (completes == true) {
                        A = nextpos;
                        parstate.x = parstate.y;
                    } else {

                    }
                }

                // particle inside [0 1]
                if (!parstate.x && parstate.y) {
                    if (completes == true) {
                        A = nextpos;
                        parstate.x = parstate.y;
                    } else {

                    }
                }
            } else {
                // update flag for next step
                flag = false;
            }

            // Store Position Data
            if (SaveAll) {
                int3 dix = make_int3(size, iter, 3);
                int3 did[4];
                did[0] = make_int3(gid, i, 0);
                did[1] = make_int3(gid, i, 1);
                did[2] = make_int3(gid, i, 2);
                did[3] = make_int3(s2i(did[0], dix), s2i(did[1], dix), s2i(did[2], dix));

                savedata[did[3].x] = A.x;
                savedata[did[3].y] = A.y;
                savedata[did[3].z] = A.z;
            }

            // Store Tensor Data
            {

                // calculate displacement
                {
                    // d2.x = fabs((A.x - xnot.x) * vsize);
                    // d2.y = fabs((A.y - xnot.y) * vsize);
                    // d2.z = fabs((A.z - xnot.z) * vsize);

                }

                diffusionTensor(&A, &xnot, vsize, dx2, dx4, &d2, i, gid, iter, size);

                // Diffusion Tensor
                {
                    /*
                    atomicAdd(&dx2[6 * i + 0], d2.x * d2.x);
                    atomicAdd(&dx2[6 * i + 1], d2.x * d2.y);
                    atomicAdd(&dx2[6 * i + 2], d2.x * d2.z);
                    atomicAdd(&dx2[6 * i + 3], d2.y * d2.y);
                    atomicAdd(&dx2[6 * i + 4], d2.y * d2.z);
                    atomicAdd(&dx2[6 * i + 5], d2.z * d2.z);
                    */
                }

                // Kurtosis Tensor
                {
                    /*
                    atomicAdd(&dx4[15 * i + 0], d2.x * d2.x * d2.x * d2.x);
                    atomicAdd(&dx4[15 * i + 1], d2.x * d2.x * d2.x * d2.y);
                    atomicAdd(&dx4[15 * i + 2], d2.x * d2.x * d2.x * d2.z);
                    atomicAdd(&dx4[15 * i + 3], d2.x * d2.x * d2.y * d2.y);
                    atomicAdd(&dx4[15 * i + 4], d2.x * d2.x * d2.y * d2.z);
                    atomicAdd(&dx4[15 * i + 5], d2.x * d2.x * d2.z * d2.z);
                    atomicAdd(&dx4[15 * i + 6], d2.x * d2.y * d2.y * d2.y);
                    atomicAdd(&dx4[15 * i + 7], d2.x * d2.y * d2.y * d2.z);
                    atomicAdd(&dx4[15 * i + 8], d2.x * d2.y * d2.z * d2.z);
                    atomicAdd(&dx4[15 * i + 9], d2.x * d2.z * d2.z * d2.z);
                    atomicAdd(&dx4[15 * i + 10], d2.y * d2.y * d2.y * d2.y);
                    atomicAdd(&dx4[15 * i + 11], d2.y * d2.y * d2.y * d2.z);
                    atomicAdd(&dx4[15 * i + 12], d2.y * d2.y * d2.z * d2.z);
                    atomicAdd(&dx4[15 * i + 13], d2.y * d2.z * d2.z * d2.z);
                    atomicAdd(&dx4[15 * i + 14], d2.z * d2.z * d2.z * d2.z);
                    */
                }

                ////////////////////////////////////////////////////////////////
                // ========================================================== //
                // ========================================================== //
                ////////////////////////////////////////////////////////////////
                ///////////////////////// SIGNAL ///////////////////////////////
                ////////////////////////////////////////////////////////////////
                // ========================================================== //
                // ========================================================== //
                ////////////////////////////////////////////////////////////////
                ////////////////////////////////////////////////////////////////

                // Signal
                /*{
                    // loop over compartments
                    s0 = 0.0;
                    for (int j = 0; j < Nc; j++) {
                        // sum over all compartments
                        s0 = s0 + (t[j] / T2[j]);
                    }

                    s0 = exp(-1.0 * s0);

                    // sig0 is the sum of all compartments
                    atomicAdd(&sig0[tidx], s0);

                    // atomAdd(&sig0[tidx], s0);

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

                    Sizes:
                    Nc: Number of Compartments
                    * t
                    * T2

                    Nbvec: Number of b values
                    * bvec
                    * bval
                    * TD

                    i: steps
                    * sigRe
                    * sig0


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
        // "/autofs/space/symphony_002/users/BenSylvanus/cuda/Sims/data"
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

    if (SaveAll) {
        printf("Executed True\n");

    } else {
        printf("Executed False\n");
    }


    for (int i = 0; i < 10; i++) {
        double value = simulationparams[i];
        simparam[i] = value;
    }
    int block_size = 256;
    int block_size = 256;
    dim3 block(block_size);
    dim3 grid((size / block.x) + 1);

    std::vector <uint64_t> bounds = sim.getbounds();
    int boundx = (int) bounds[0];
    int boundy = (int) bounds[1];
    int boundz = (int) bounds[2];
    int prod = (int) (boundx * boundy * boundz);
    std::vector<double> r_swc = sim.getSwc();
    int nrow = r_swc.size() / 6;

    //old comments
    {
        /** 
         * @brief Simulation Params Array
         * <li> particle_num = SimulationParams[0] </li>
         * <li> step_num = SimulationParams[1] </li>
         * <li> step_size = SimulationParams[2] </li>
         * <li> perm_prob = SimulationParams[3] </li>
         * <li> init_in = SimulationParams[4] </li>
         * <li> D0 = SimulationParams[5] </li>
         * <li> d = SimulationParams[6] </li>
         * <li> scale = SimulationParams[7] </li>
         * <li> tstep = SimulationParams[8] </li>
         * <li> vsize = SimulationParams[9] </li>
         * @brief INDEXING SWC ARRAY
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
        // stride + bx * (by * y + z) + x
        // int id0 = 0 + (boundx) * ((boundy) * 2 + 2) + 3;
        // printf("lut[%d]: %d\n", id0, lut[id0]);
        // ----------------------
        // Lookup Table Summary
        // linearindex = stride + bx * (by * z + y) + x
        // voxel coord: (x,y,z);
        // ----------------------
    }

    double4 swc_trim[nrow];
    double w_swc[nrow * 4];
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

    /**
     * Host Section:
     * - Create Pointers
     * - Allocate Memory
     * - Set Values
     */
    // Create Host Pointers
    int *hostBounds;
    double *hostdx2;
    double *hostdx4;
    double *hostSimP;
    int *hostNewLut;
    int *hostNewIndex;
    int *hostIndexSize;
    double4 *hostD4Swc;
    double *mdx2;
    double *mdx4;
    double *hostAllData;
    double *hostReflections;
    double *hosturef;
    int* hostFlip;

    // signal variables
    /*
    Sizes:
    Nc: Number of Compartments
    * t * 1
    * T2 * 1

    Nbvec: Number of b values
    * bvec * 3
    * bval * 1
    * TD * 1

    i: steps
    * sigRe Nbvec * i
    * sig0 Nc * i
    *

    // sum of all signals per time step per b value

    &sigRe[Nbvec * tidx + j], s0 * cos(qx));
    */

    double *hostT2; // Nc * 1
    double *hostT; // Nc * 1
    double *hostSigRe; // Nbvec * iter
    double *hostSig0; // Nc * iter
    double *hostbvec; // Nbvec * 3 (x,y,z)
    double *hostbval; // Nbvec * 1 (b)
    double *hostTD; // Nbvec * 1 (TD)



    // Alloc Memory for Host Pointers
    {

        hostBounds = (int *) malloc(3 * SOI);
        hostdx2 = (double *) malloc(6 * iter * SOD);
        hostdx4 = (double *) malloc(15 * iter * SOD);
        hostSimP = (double *) malloc(10 * SOD);
        hostD4Swc = (double4 *) malloc(nrow * SOD4);
        hostNewLut = (int *) malloc(prod * SOI);
        hostNewIndex = (int *) malloc(newindexsize * SOI);
        hostIndexSize = (int *) malloc(3 * SOI);
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

        // signal variables
        hostT2 = (double *) malloc(Nc * SOD);
        hostT = (double *) malloc(Nc * SOD);
        hostSigRe = (double *) malloc(Nbvec * iter * SOD);
        hostSig0 = (double *) malloc(Nc * iter * SOD);
        hostbvec = (double *) malloc(Nbvec * 3 * SOD);
        hostbval = (double *) malloc(Nbvec * SOD);
        hostTD = (double *) malloc(Nbvec * SOD);

        printf("Allocated Host Data\n");
    }

    // Set Values for Host
    {

        hostBounds[0] = boundx;
        hostBounds[1] = boundy;
        hostBounds[2] = boundz;
        memset(hostdx2, 0.0, 6 * iter * SOD);
        memset(hostdx4, 0.0, 15 * iter * SOD);
        {

            for (int i = 0; i < 10; i++) {
                hostSimP[i] = simparam[i];
            }
    {

        hostBounds[0] = boundx;
        hostBounds[1] = boundy;
        hostBounds[2] = boundz;
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

            for (int i = 0; i < 3; i++) {
                int value = index_dims[i];
                hostIndexSize[i] = value;
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

        memset(hostSigRe, 0.0, Nbvec * iter * SOD); // Calculated in kernel
        memset(hostSig0, 0.0, Nc * iter * SOD); // Calculated in kernel
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
    int *deviceBounds;
    double *deviceSimP;
    double4 *deviced4Swc;
    int *deviceNewLut;
    int *deviceNewIndex;
    int *deviceIndexSize;
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
        gpuErrchk(cudaMalloc((int **) &deviceBounds, 3 * SOI));
        gpuErrchk(cudaMalloc((curandStatePhilox4_32_10_t * *) & deviceState, size * sizeof(curandStatePhilox4_32_10_t)));
        gpuErrchk(cudaMalloc((double **) &deviceSimP, 10 * SOD));
        gpuErrchk(cudaMalloc((double4 * *) & deviced4Swc, nrow * SOD4));
        gpuErrchk(cudaMalloc((int **) &deviceNewLut, prod * SOI));
        gpuErrchk(cudaMalloc((int **) &deviceNewIndex, newindexsize * SOI));
        gpuErrchk(cudaMalloc((int **) &deviceIndexSize, 3 * SOI));
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

        gpuErrchk(cudaMalloc((double **) &deviceSigRe, Nbvec * iter * SOD));
        gpuErrchk(cudaMalloc((double **) &deviceSig0, Nc * iter * SOD));
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
        gpuErrchk(cudaMemcpy(deviceBounds, hostBounds, 3 * SOI, cudaMemcpyHostToDevice));

        setup_kernel<<<grid, block>>>(deviceState, 1);
        
        gpuErrchk(cudaMemcpy(deviceSimP, hostSimP, 10 * SOD, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(deviced4Swc, hostD4Swc, nrow * SOD4, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(deviceNewLut, hostNewLut, prod * SOI, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(deviceNewIndex, hostNewIndex, newindexsize * SOI, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(deviceIndexSize, hostIndexSize, 3 * SOI, cudaMemcpyHostToDevice));
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

        gpuErrchk(cudaMemcpy(deviceSigRe, hostSigRe, Nbvec * iter * SOD, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(deviceSig0, hostSig0, Nc * iter * SOD, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(devicebvec, hostbvec, Nbvec * 3 * SOD, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(devicebval, hostbval, Nbvec * SOD, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(deviceTD, hostTD, Nbvec * SOD, cudaMemcpyHostToDevice));

    }

    // option for printing in kernel
    bool debug = false;
    double3 point = make_double3(hostD4Swc[0].x, hostD4Swc[0].y, hostD4Swc[0].z);

    double3 point = make_double3(hostD4Swc[0].x, hostD4Swc[0].y, hostD4Swc[0].z);

    /**
     * Call Kernel
    */
    printf("Simulating...\n");

    // kernel
    {

        simulate<<<grid, block>>>(deviceAllData, devicedx2, devicedx4, deviceBounds, deviceState, deviceSimP,
                                  deviced4Swc, deviceNewLut, deviceNewIndex, deviceIndexSize, size, iter, debug, point,
                                  SaveAll,
                                  deviceReflections, deviceURef, deviceFlip, // reflection variables
                                  deviceT2, deviceT, deviceSigRe, deviceSig0, devicebvec, devicebval, deviceTD); // signal variables
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
        gpuErrchk(cudaMemcpy(hostSigRe, deviceSigRe, Nbvec * iter * SOD, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(hostSig0, deviceSig0, Nc * iter * SOD, cudaMemcpyDeviceToHost));
    }

    cudaEventSynchronize(stop_c);
    cudaEventElapsedTime(&milliseconds, start_c, stop_c);
    end = clock();
    printf("Kernel took %f seconds\n", milliseconds / 1e3);
    printf("Kernel took %f seconds\n", milliseconds / 1e3);
    auto t1 = high_resolution_clock::now();

    // Free Device Memory
    {

        printf("Freeing Device Data: ");
        gpuErrchk(cudaFree(deviceBounds));
        gpuErrchk(cudaFree(deviceState));
        gpuErrchk(cudaFree(devicedx2));
        gpuErrchk(cudaFree(devicedx4));
        gpuErrchk(cudaFree(deviceSimP));
        gpuErrchk(cudaFree(deviced4Swc));
        gpuErrchk(cudaFree(deviceNewLut));
        gpuErrchk(cudaFree(deviceNewIndex));
        gpuErrchk(cudaFree(deviceIndexSize));
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


    // Free Device Memory
    {

        printf("Freeing Device Data: ");
        cudaFree(deviceBounds);
        cudaFree(deviceState);
        cudaFree(devicedx2);
        cudaFree(devicedx4);
        cudaFree(deviceSimP);
        cudaFree(deviced4Swc);
        cudaFree(deviceNewLut);
        cudaFree(deviceNewIndex);
        cudaFree(deviceIndexSize);
        cudaFree(deviceAllData);
        // Reflection Variables
        cudaFree(deviceReflections);
        cudaFree(deviceURef);
        cudaFree(deviceFlip);
        // Signal Variables
        cudaFree(deviceT2);
        cudaFree(deviceT);
        cudaFree(deviceSigRe);
        cudaFree(deviceSig0);
        cudaFree(devicebvec);
        cudaFree(devicebval);
        cudaFree(deviceTD);

    }

    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    printf("%f seconds\n", ms_double.count() / 1e3);
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
        fwrite(hostSig0, SOD, Nc * iter, outFile);
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
    printf("%f seconds\n", ms_double.count() / 1e3);

    // Free Host Memory
    {

        free(hostBounds);
        free(hostdx2);
        free(hostdx4);
        free(hostSimP);
        free(hostD4Swc);
        free(hostNewIndex);
        free(hostIndexSize);
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
