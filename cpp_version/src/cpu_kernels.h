#pragma once

#include <vector>
#include "cuda_replacements.h"
#include <cstdint>
// Constants
constexpr int Nc = 2;

void volfrac_cpu(const std::vector<double4>& d4swc, const std::vector<int>& nlut, const std::vector<int>& NewIndex,
                 int3 Bounds, int3 IndexSize, int n, std::vector<int>& label, double& vf, bool debug);

void simulate_cpu(std::vector<double>& savedata, std::vector<double>& dx2, std::vector<double>& dx4, 
                  int3 Bounds, const std::vector<double>& SimulationParams, const std::vector<double4>& d4swc, 
                  const std::vector<int>& nlut, const std::vector<int>& NewIndex, int3 IndexSize, int size, int iter, 
                  bool debug, double3 point, int SaveAll, std::vector<double>& Reflections, 
                  std::vector<double>& Uref, std::vector<int>& flip, const std::vector<double>& T2, 
                  std::vector<double>& T, std::vector<double>& Sig0, std::vector<double>& SigRe, 
                  const std::vector<double>& BVec, const std::vector<double>& BVal, const std::vector<double>& TD,
                  int num_threads);




