//
// Created by Benjamin Sylvanus on 2/28/23.
//
#include "cpu_functions.h"

void simulate_cpu(double *ip, double *psr, double *psn, double *dx2, double *Bounds, int *LUT,
             int *IndexArray, double *SWC, int size, const int pairmax, int iter) {

    /**
     * @ip
     * type: (double *) \n
     * size: [npar * 3];
     * @psr
     * type: (double *) \n
     * size: [npar * nstep * 4];
     * @psn
     * type: (double *) \n
     * size: [npar * nstep * 3];
     */

    for (int gid = 0; gid < size; ++gid) {
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
        /**
         * instead of generating a random double4 we read the values from state[gid];
         * <li> init local state var </li>
         *  curandStatePhilox4_32_10_t localstate = state[gid] -> none;
         * <li> set particle initial position</li>
         *  xi = curand_uniform4_double(&localstate) -> read state[gid]
         */

        /**
         * First step is to verify the random position initialization
         * <li>  The chosed random position for this process </li>
         * <li> Confirm the position chosen is inside our geometry </li>
         */

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



void initialize_cpu(int npar, double *gpuPositions, double *Bounds, int *LUT, int *IndexArray,
                    double *SWC, const int pairmax) {

    double bx = Bounds[0];
    double by = Bounds[1];
    bool positionValid;
    bool inside;

    for (int i = 0; i < npar; i++) {
        positionValid = false;
        // check whether position of ith particle is inside.
        double xi[3];
        int fxi[3];
        int lxi[3];
        int hxi[3];

        for (int j = 0; j < 3; j++) {
            xi[j] = gpuPositions[3 * i + j];
            fxi[j] = (int) gpuPositions[3 * i + j];
            lxi[j] = fxi[j] < Bounds[j];
            hxi[j] = fxi[j] > 0;
        }

        // if floor of position is inside bounds of lookup table
        if (lxi[0] && lxi[1] && lxi[2] && hxi[0] && hxi[1] && hxi[2]) {
            // calculate index of lookup table
            int ixlut = 0 + bx * (by * fxi[0] + fxi[1]) + fxi[2];

            // get value at linear index
            int volut = LUT[ixlut];

            // check all pairs in index array
            for (int j = 0; j < pairmax; j++) {
                // calculate index in index array
                int cix_index = pairmax * volut + (2 * j + 0);
                int pix_index = pairmax * volut + (2 * j + 1);

                // extract ids from index array
                int cix = IndexArray[cix_index];
                int pix = IndexArray[pix_index];

                // child values
                double cv_swc[4];

                // parent values
                double pv_swc[4];

                for (int k = 0; k < 4; k++) {
                    // child value(k) -> swc (4 * child_index + k)
                    cv_swc[k] = SWC[4 * cix + k];
                    pv_swc[k] = SWC[4 * pix + k];
                }

                inside = swc2v_cpu(xi, cv_swc, pv_swc);
                if (inside) {
                    positionValid = true;
                    j = pairmax;
                }
            }
            if (!positionValid) {
                printf("Position(%d) failed inside calculation.\n", i);
                return;
            }
        } else {
            positionValid = false;
            printf("Particle(%d) failed bounds check.\n", i);
            return;
        }
    }
    printf("All particles have valid initial positions.\n");
    return;
}

