//
// Created by Benjamin Sylvanus on 2/28/23.
//

#ifndef CUDAPREP_CPU_FUNCTIONS_H
#define CUDAPREP_CPU_FUNCTIONS_H

/**
 *
 * @param ip initial positions of all particles
 * @param psr random values for each particle for each step
 * @param psn positions for each particle each step
 * @param dx2 mean displacement vector for each step
 * @param Bounds bounds of Lookup Table
 * @param LUT Lookup Table of simulation
 * @param IndexArray Index array of simulation
 * @param SWC SWC of simulation
 * @param size number of particles to simulate
 * @param pairmax max number of pairs in index array, used as index scalar
 * @param iter number of steps to simulate
 */
void simulate_cpu(double *ip, double *psr, double *psn, double *dx2, double *Bounds, int *LUT, int *IndexArray,
                  double *SWC, int size, const int pairmax, int iter);


/**
 * @param npar number of particles to simulate: alias size.
 * @param gpuPositions positions of each particle:
 * @param positionValid array for result of each particles initial position:
 * @param Bounds bounds of our geometry
 * @param LUT lookuptable of simulation
 * @param IndexArray index array of simulation
 * @param SWC swc of simulation
 * @param pairmax max number of pairs in index array
 * @indexing npar none \n
 * @indexing gpuPositions X: [3 * i + 0] Y:[3 * i + 1] Z: [3 * i + 2]  \n
 * @indexing positionValid particle: [i] \n
 * @indexing Bounds X: [0]  Y: [1]  Z: [2] \n
 * @indexing LUT linear index: 0 + dx * (dy * x + y) + z; \n
 * @indexing IndexArray j:E(0:pairmax), [pairmax * i + (2 * j + 0)] [pairmax * i + (2 * j + 1)] \n
 * @indexing SWC X: [4*i+0]  Y: [4*i+1]  Z: [4*i+2]  Radius: [4*i+3] \n
 * @indexing pairmax none \n
 */
void initialize_cpu(int npar, double *gpuPositions, double *Bounds, int *LUT, int *IndexArray,
                    double *SWC, const int pairmax);


#endif //CUDAPREP_CPU_FUNCTIONS_H
