//
// Created by Benjamin Sylvanus on 6/5/23.
//

#ifndef CUDAPREP_U_OVERLOADS_H
#define CUDAPREP_U_OVERLOADS_H

#include "simreader.h"
#include "simulation.h"
#include "controller.h"
#include "viewer.h"
#include "funcs.h"
#include "overloads.h"

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

//////////////////////////////////////////
// + operator overloads
//////////////////////////////////////////
__device__ double3 operator+(const double3 &lhs, const double3 &rhs);

__device__ double3 operator+(const double3 &lhs, const double &rhs);

__device__ double3 operator+(const double &lhs, const double3 &rhs);

__device__ double3 operator+=(double3 &lhs, const double3 &rhs);

__device__ double3 operator+=(double3 &lhs, const double &rhs);

//////////////////////////////////////////
// - operator overloads
//////////////////////////////////////////
__device__ double3 operator-(const double3 &lhs, const double3 &rhs);

__device__ double3 operator-(const double3 &lhs, const double &rhs);

__device__ double3 operator-(const double &lhs, const double3 &rhs);

__device__ double3 operator-=(double3 &lhs, const double3 &rhs);

__device__ double3 operator-=(double3 &lhs, const double &rhs);

//////////////////////////////////////////
// * operator overloads
//////////////////////////////////////////
__device__ double3 operator*(const double3 &lhs, const double3 &rhs);

__device__ double3 operator*(const double3 &lhs, const double &rhs);

__device__ double3 operator*(const double &lhs, const double3 &rhs);

__device__ double3 operator*=(double3 &lhs, const double3 &rhs);

__device__ double3 operator*=(double3 &lhs, const double &rhs);

//////////////////////////////////////////
// / operator overloads
//////////////////////////////////////////

__device__ double3 operator/(const double3 &lhs, const double3 &rhs);

__device__ double3 operator/(const double3 &lhs, const double &rhs);

__device__ double3 operator/(const double &lhs, const double3 &rhs);

__device__ double3 operator/=(double3 &lhs, const double3 &rhs);

__device__ double3 operator/=(double3 &lhs, const double &rhs);

//////////////////////////////////////////
// > operator overloads
//////////////////////////////////////////

__device__ int3 operator>(const double3 &lhs, const double3 &rhs);

//////////////////////////////////////////
// < operator overloads
//////////////////////////////////////////

__device__ int3 operator<(const double3 &lhs, const double3 &rhs);

//////////////////////////////////////////
// dot product
//////////////////////////////////////////

__device__ double dot(const double3 &lhs, const double3 &rhs);

//////////////////////////////////////////
// length
//////////////////////////////////////////

__device__ double length(const double3 &lhs);

//////////////////////////////////////////
// normalize
//////////////////////////////////////////

__device__ double3 normalize(const double3 &lhs);

//////////////////////////////////////////
// distance: double3
//////////////////////////////////////////

__device__ double distance2(const double3 &lhs, const double3 &rhs);

//////////////////////////////////////////
// distance: double4
//////////////////////////////////////////

__device__ double distance2(const double4 &lhs, const double4 &rhs);

//////////////////////////////////////////
// set (double &lhs, int3 &index, double3 &values);
//////////////////////////////////////////

__device__ void set(double *lhs, int3 &index, double3 &values);





#endif //CUDAPREP_U_OVERLOADS_H
