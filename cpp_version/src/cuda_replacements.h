#ifndef CUDA_REPLACEMENTS_H
#define CUDA_REPLACEMENTS_H

#include <cmath>

// Type definitions
struct int2 { int x, y; };
struct int3 { int x, y, z; };
struct double3 { double x, y, z; };
struct double4 { double x, y, z, w; };

// Helper functions
inline int2 make_int2(int x, int y) { int2 result = {x, y}; return result; }
inline int3 make_int3(int x, int y, int z) { int3 result = {x, y, z}; return result; }
inline double3 make_double3(double x, double y, double z) { double3 result = {x, y, z}; return result; }
inline double4 make_double4(double x, double y, double z, double w) { double4 result = {x, y, z, w}; return result; }

// Constants
const int Nbvec = 3;
const int timepoints = 100; // Adjust this value as needed

// Replace __device__ with inline for C++
#define __device__ inline

#endif // CUDA_REPLACEMENTS_H