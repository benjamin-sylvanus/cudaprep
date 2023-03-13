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

__device__ bool swc2v(double3 nextpos, double4 child, double4 parent, double dist) {
    /**
    * @brief calculation for tangent line
    * <li> r = r1 + sqrt((x-x1).^2 + (y-y1).^2 + (z-z1).^2) / sqrt((x2-x1)^2+(parent.y-y1)^2+(parent.z-z1)^2) * (r2-r1) </li>
    * <li> r = ( c + r2 ) / (sqrt ( 1 - ( |r1-r2 | / l ) ) </li>
    * <li>c = ( |r1 - r2| * l ) / L </li>
   */
    bool pos1;
    bool pos;

    if (dist == 0) {
        dist = 0.000000000000000001;
    }

    double t = ((nextpos.x - child.x) * (parent.x - child.x) + (nextpos.y - child.y) * (parent.y - child.y) +
                (nextpos.z - child.z) * (parent.z - child.z)) / dist;
    double x = child.x + (parent.x - child.x) * t;
    double y = child.y + (parent.y - child.y) * t;
    double z = child.z + (parent.z - child.z) * t;
    double cr2 = pow(child.w, 2);
    double pr2 = pow(parent.w, 2);
    double rd = fabs(child.w - parent.w);

    bool list1;
    if (dist < cr2) {
        list1 = false;
    } else {
        list1 = (x - child.x) * (x - parent.x) + (y - child.y) * (y - parent.y) + (z - child.z) * (z - parent.z) < 0.0;
    }

    if (list1) {
        double dist2 = (pow(nextpos.x - x, 2) + pow(nextpos.y - y, 2) + pow(nextpos.z - z, 2));

        // distance from orthogonal vector to p2
        double l = sqrt(pow(x - parent.x, 2) + pow(y - parent.y, 2) + pow(z - parent.z, 2));

        // distance from p1 -> p2
        double L = sqrt(dist);

        double c = (rd * l) / L;
        double r = (c + parent.w) / sqrt(1 - ((rd / L) * (rd / L)));
        pos1 = dist2 < (pow(r, 2));
        pos = pos1;
    } else {
        double lower = (pow(nextpos.x - child.x, 2) + pow(nextpos.y - child.y, 2) + pow(nextpos.z - child.z, 2));
        double higher = (pow(nextpos.x - parent.x, 2) + pow(nextpos.y - parent.y, 2) + pow(nextpos.z - parent.z, 2));
        pos1 = (pow(nextpos.x - child.x, 2) + pow(nextpos.y - child.y, 2) + pow(nextpos.z - child.z, 2)) < cr2 ||
               (pow(nextpos.x - parent.x, 2) + pow(nextpos.y - parent.y, 2) + pow(nextpos.z - parent.z, 2)) < pr2;
        pos = pos1;
    }
    return pos;
}


__device__ __host__ int s2i(int3 i, int3 b) {
    return 0 + b.x * (b.y * i.z + i.y) + i.x;
}

__device__ double3 initPosition(int gid, double *dx2, int *Bounds, curandStatePhilox4_32_10_t *state,
                    double *SimulationParams, double4 *d4swc, int *nlut, int *NewIndex,
                    int *IndexSize, int size, int iter, bool debug) {

    double3 nextpos;
    int3 upper;
    int3 lower;
    int3 floorpos;
    int3 b_int3 = make_int3(Bounds[0], Bounds[1], Bounds[2]);
    int3 i_int3 = make_int3(IndexSize[0], IndexSize[1], IndexSize[2]);
    double3 b_d3 = make_double3((double) b_int3.x, (double) b_int3.y, (double) b_int3.z);
    curandStatePhilox4_32_10_t localstate = state[gid];
    double4 xr;
    int2 parstate;
    double3 A;
    parstate.y = 0;
    bool cont = true;
    while (cont) {

        // init local state var
        xr = curand_uniform4_double(&localstate);

        // set particle initial position
        A = make_double3(xr.x * b_d3.x, xr.y * b_d3.y, xr.z * b_d3.z);

        nextpos = A;

        // floor of position -> check voxels
        floorpos.x = (int) A.x;
        floorpos.y = (int) A.y;
        floorpos.z = (int) A.z;

        // upper bounds of lookup table
        upper.x = floorpos.x < (Bounds[0] - 0);
        upper.y = floorpos.y < (Bounds[1] - 0);
        upper.z = floorpos.z < (Bounds[2] - 0);

        // lower bounds of lookup table
        lower.x = floorpos.x >= 0;
        lower.y = floorpos.y >= 0;
        lower.z = floorpos.z >= 0;

        // position inside the bounds of volume -> state of next position true : false
        parstate.x = (lower.x && lower.y && lower.z && upper.x && upper.y && upper.z) ? 1 : 0;

        double4 parent;
        double4 child;
        int2 vindex;
        int id_test = s2i(floorpos, b_int3);
        int test_lutvalue = nlut[id_test];
        double dist2;
        if (parstate.x) {
            for (int page = 0; page < i_int3.z; page++) {
                int3 c_new = make_int3(test_lutvalue, 0, page);
                int3 p_new = make_int3(test_lutvalue, 1, page);
                vindex.x = NewIndex[s2i(c_new, i_int3)] - 1;
                vindex.y = NewIndex[s2i(p_new, i_int3)] - 1;

                if ((vindex.x) != -1) {
                    child = make_double4(d4swc[vindex.x].x, d4swc[vindex.x].y, d4swc[vindex.x].z, d4swc[vindex.x].w);
                    parent = make_double4(d4swc[vindex.y].x, d4swc[vindex.y].y, d4swc[vindex.y].z, d4swc[vindex.y].w);

                    //distance squared between child parent
                    dist2 = pow(parent.x - child.x, 2) + pow(parent.y - child.y, 2) + pow(parent.z - child.z, 2);

                    // determine whether particle is inside this connection
                    bool inside = swc2v(nextpos, child, parent, dist2);

                    // if it is inside the connection we don't need to check the remaining.
                    if (inside) {

                        // update the particles state
                        parstate.y = 1;
                        // end for p loop
                        page = i_int3.z;
                        cont = false;
                    }
                }
            }
        }
    }
    return A;
}

__device__ void diffusionTensor(double3 A, double3 xnot, double vsize, double *dx2, double * savedata, double3 d2, int i, int gid, int iter, int size) {

    d2.x = fabs((A.x - xnot.x) * vsize);
    d2.y = fabs((A.y - xnot.y) * vsize);
    d2.z = fabs((A.z - xnot.z) * vsize);

    // diffusion tensor
    atomicAdd(&dx2[6 * i + 0], d2.x * d2.x);
    atomicAdd(&dx2[6 * i + 1], d2.x * d2.y);
    atomicAdd(&dx2[6 * i + 2], d2.x * d2.z);
    atomicAdd(&dx2[6 * i + 3], d2.y * d2.y);
    atomicAdd(&dx2[6 * i + 4], d2.y * d2.z);
    atomicAdd(&dx2[6 * i + 5], d2.z * d2.z);
    printf("dx2[6*i+0]%f \n",dx2[6+i+0]);

    int3 dix = make_int3(iter, size, 3);
    int3 did[4];
    did[0] = make_int3(0, i, gid);
    did[1] = make_int3(1, i, gid);
    did[2] = make_int3(2, i, gid);
    did[3] = make_int3(s2i(did[0], dix), s2i(did[1], dix), s2i(did[2], dix));
    savedata[did[3].x] = A.x;
    savedata[did[3].y] = A.y;
    savedata[did[3].z] = A.z;
}


__device__ double3 setNextPos(double3 nextpos, double3 A, double4 xi, double step)
{
    nextpos.x = A.x + ((2.0 * xi.x - 1.0) * step);
    nextpos.y = A.y + ((2.0 * xi.y - 1.0) * step);
    nextpos.z = A.z + ((2.0 * xi.z - 1.0) * step);
    return nextpos;
}
