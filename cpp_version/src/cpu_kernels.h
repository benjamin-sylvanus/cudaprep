#pragma once

#include <vector>
#include "cuda_replacements.h"

// Constants
const int Nc = 2;

// Note: Nbvec and timepoints are already defined in cuda_replacements.h

/**
 *
 * @param d4swc
 * @param nlut
 * @param NewIndex
 * @param Bounds
 * @param IndexSize
 * @param n
 * @param label
 * @param vf
 */
void volfrac_cpu(const std::vector<double4>& d4swc, const std::vector<int>& nlut, const std::vector<int>& NewIndex,
                 int3 Bounds, int3 IndexSize, int n, std::vector<int>& label, double& vf);

/**
 *
 * @param savedata
 * @param dx2
 * @param dx4
 * @param Bounds
 * @param SimulationParams
 * @param d4swc
 * @param nlut
 * @param NewIndex
 * @param IndexSize
 * @param size
 * @param iter
 * @param debug
 * @param point
 * @param SaveAll
 * @param Reflections
 * @param Uref
 * @param flip
 * @param T2
 * @param T
 * @param Sig0
 * @param SigRe
 * @param BVec
 * @param BVal
 * @param TD
 */
void simulate_cpu(std::vector<double> &savedata, std::vector<double> &dx2, std::vector<double> &dx4,
                  int3 Bounds, const std::vector<double> &SimulationParams, const std::vector<double4> &d4swc,
                  const std::vector<int> &nlut, const std::vector<int> &NewIndex, int3 IndexSize, int size, int iter,
                  bool debug, double3 point, int SaveAll, std::vector<double> &Reflections,
                  std::vector<double> &Uref, std::vector<int> &flip, const std::vector<double> &T2,
                  std::vector<double> &T, std::vector<double> &Sig0, std::vector<double> &SigRe,
                  const std::vector<double> &BVec, const std::vector<double> &BVal,
                  const std::vector<double> &TD);

void validCoord(double3 &nextpos, double3 &pos, int3 &b_int3, int3 &upper, int3 &lower, int3 &floorpos,
                double *reflections, double *uref, int gid, int i, int size, int iter, int *flips);
void setData(double* data, int3 idx, double3 value);
double distance2(const double4 &lhs, const double4 &rhs);