//
// Created by Benjamin Sylvanus on 6/5/23.
//

#ifndef CUDAPREP_U_OVERLOADS_H
#define CUDAPREP_U_OVERLOADS_H

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

//////////////////////////////////////////
// + operator overloads
//////////////////////////////////////////
double3 operator+(const double3 &lhs, const double3 &rhs) {
    return make_double3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

double3 operator+(const double3 &lhs, const double &rhs) {
    return make_double3(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs);
}

double3 operator+(const double &lhs, const double3 &rhs) {
    return make_double3(lhs + rhs.x, lhs + rhs.y, lhs + rhs.z);
}

double3 operator+=(double3 &lhs, const double3 &rhs) {
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    return lhs;
}

double3 operator+=(double3 &lhs, const double &rhs) {
    lhs.x += rhs;
    lhs.y += rhs;
    lhs.z += rhs;
    return lhs;
}

//////////////////////////////////////////
// - operator overloads
//////////////////////////////////////////
double3 operator-(const double3 &lhs, const double3 &rhs) {
    return make_double3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

double3 operator-(const double3 &lhs, const double &rhs) {
    return make_double3(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs);
}

double3 operator-(const double &lhs, const double3 &rhs) {
    return make_double3(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z);
}

double3 operator-=(double3 &lhs, const double3 &rhs) {
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
    return lhs;
}

double3 operator-=(double3 &lhs, const double &rhs) {
    lhs.x -= rhs;
    lhs.y -= rhs;
    lhs.z -= rhs;
    return lhs;
}

//////////////////////////////////////////
// * operator overloads
//////////////////////////////////////////
double3 operator*(const double3 &lhs, const double3 &rhs) {
    return make_double3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z);
}

double3 operator*(const double3 &lhs, const double &rhs) {
    return make_double3(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
}

double3 operator*(const double &lhs, const double3 &rhs) {
    return make_double3(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z);
}

double3 operator*=(double3 &lhs, const double3 &rhs) {
    lhs.x *= rhs.x;
    lhs.y *= rhs.y;
    lhs.z *= rhs.z;
    return lhs;
}

double3 operator*=(double3 &lhs, const double &rhs) {
    lhs.x *= rhs;
    lhs.y *= rhs;
    lhs.z *= rhs;
    return lhs;
}

//////////////////////////////////////////
// / operator overloads
//////////////////////////////////////////

double3 operator/(const double3 &lhs, const double3 &rhs) {
    return make_double3(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z);
}

double3 operator/(const double3 &lhs, const double &rhs) {
    return make_double3(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs);
}

double3 operator/(const double &lhs, const double3 &rhs) {
    return make_double3(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z);
}

double3 operator/=(double3 &lhs, const double3 &rhs) {
    lhs.x /= rhs.x;
    lhs.y /= rhs.y;
    lhs.z /= rhs.z;
    return lhs;
}

double3 operator/=(double3 &lhs, const double &rhs) {
    lhs.x /= rhs;
    lhs.y /= rhs;
    lhs.z /= rhs;
    return lhs;
}

//////////////////////////////////////////
// == operator overloads
//////////////////////////////////////////
bool operator==(const double3 &lhs, const double3 &rhs) {
    return (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z);
}

bool operator==(const double3 &lhs, const double &rhs) {
    return (lhs.x == rhs && lhs.y == rhs && lhs.z == rhs);
}

bool operator==(const double &lhs, const double3 &rhs) {
    return (lhs == rhs.x && lhs == rhs.y && lhs == rhs.z);
}

//////////////////////////////////////////
// != operator overloads
//////////////////////////////////////////

bool operator!=(const double3 &lhs, const double3 &rhs) {
    return (lhs.x != rhs.x || lhs.y != rhs.y || lhs.z != rhs.z);
}

bool operator!=(const double3 &lhs, const double &rhs) {
    return (lhs.x != rhs || lhs.y != rhs || lhs.z != rhs);
}

bool operator!=(const double &lhs, const double3 &rhs) {
    return (lhs != rhs.x || lhs != rhs.y || lhs != rhs.z);
}

//////////////////////////////////////////
// < operator overloads
//////////////////////////////////////////

bool operator<(const double3 &lhs, const double3 &rhs) {
    return (lhs.x < rhs.x && lhs.y < rhs.y && lhs.z < rhs.z);
}

bool operator<(const double3 &lhs, const double &rhs) {
    return (lhs.x < rhs && lhs.y < rhs && lhs.z < rhs);
}

bool operator<(const double &lhs, const double3 &rhs) {
    return (lhs < rhs.x && lhs < rhs.y && lhs < rhs.z);
}

//////////////////////////////////////////
// > operator overloads
//////////////////////////////////////////

bool operator>(const double3 &lhs, const double3 &rhs) {
    return (lhs.x > rhs.x && lhs.y > rhs.y && lhs.z > rhs.z);
}

bool operator>(const double3 &lhs, const double &rhs) {
    return (lhs.x > rhs && lhs.y > rhs && lhs.z > rhs);
}

bool operator>(const double &lhs, const double3 &rhs) {
    return (lhs > rhs.x && lhs > rhs.y && lhs > rhs.z);
}

//////////////////////////////////////////
// <= operator overloads
//////////////////////////////////////////

bool operator<=(const double3 &lhs, const double3 &rhs) {
    return (lhs.x <= rhs.x && lhs.y <= rhs.y && lhs.z <= rhs.z);
}

bool operator<=(const double3 &lhs, const double &rhs) {
    return (lhs.x <= rhs && lhs.y <= rhs && lhs.z <= rhs);
}

bool operator<=(const double &lhs, const double3 &rhs) {
    return (lhs <= rhs.x && lhs <= rhs.y && lhs <= rhs.z);
}

//////////////////////////////////////////
// >= operator overloads
//////////////////////////////////////////

bool operator>=(const double3 &lhs, const double3 &rhs) {
    return (lhs.x >= rhs.x && lhs.y >= rhs.y && lhs.z >= rhs.z);
}

bool operator>=(const double3 &lhs, const double &rhs) {
    return (lhs.x >= rhs && lhs.y >= rhs && lhs.z >= rhs);
}

bool operator>=(const double &lhs, const double3 &rhs) {
    return (lhs >= rhs.x && lhs >= rhs.y && lhs >= rhs.z);
}

double3 operator-(const double3 &lhs) {
    return make_double3(-lhs.x, -lhs.y, -lhs.z);
}


//////////////////////////////////////////
// dot product
//////////////////////////////////////////

double dot(const double3 &lhs, const double3 &rhs) {
    return (lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z);
}

//////////////////////////////////////////
// cross product
//////////////////////////////////////////

double3 cross(const double3 &lhs, const double3 &rhs) {
    return make_double3(lhs.y * rhs.z - lhs.z * rhs.y,
                        lhs.z * rhs.x - lhs.x * rhs.z,
                        lhs.x * rhs.y - lhs.y * rhs.x);
}

//////////////////////////////////////////
// length
//////////////////////////////////////////

double length(const double3 &lhs) {
    return sqrt(lhs.x * lhs.x + lhs.y * lhs.y + lhs.z * lhs.z);
}

//////////////////////////////////////////
// normalize
//////////////////////////////////////////

double3 normalize(const double3 &lhs) {
    double invLen = 1.0 / sqrt(lhs.x * lhs.x + lhs.y * lhs.y + lhs.z * lhs.z);
    return make_double3(lhs.x * invLen, lhs.y * invLen, lhs.z * invLen);
}

//////////////////////////////////////////
// floor
//////////////////////////////////////////

double3 floor(const double3 &lhs) {
    return make_double3(floor(lhs.x), floor(lhs.y), floor(lhs.z));
}













#endif //CUDAPREP_U_OVERLOADS_H
