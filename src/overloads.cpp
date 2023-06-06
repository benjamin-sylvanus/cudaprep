//
// Created by Benjamin Sylvanus on 6/5/23.
//

#include "overloads.h"

//////////////////////////////////////////
// + operator overloads
//////////////////////////////////////////
__device__ double3 operator+(const double3 &lhs, const double3 &rhs) {
    return make_double3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

__device__ double3 operator+(const double3 &lhs, const double &rhs) {
    return make_double3(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs);
}

__device__ double3 operator+(const double &lhs, const double3 &rhs) {
    return make_double3(lhs + rhs.x, lhs + rhs.y, lhs + rhs.z);
}

__device__ double3 operator+=(double3 &lhs, const double3 &rhs) {
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    return lhs;
}

__device__ double3 operator+=(double3 &lhs, const double &rhs) {
    lhs.x += rhs;
    lhs.y += rhs;
    lhs.z += rhs;
    return lhs;
}

//////////////////////////////////////////
// - operator overloads
//////////////////////////////////////////
__device__ double3 operator-(const double3 &lhs, const double3 &rhs) {
    return make_double3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

__device__ double3 operator-(const double3 &lhs, const double &rhs) {
    return make_double3(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs);
}

__device__ double3 operator-(const double &lhs, const double3 &rhs) {
    return make_double3(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z);
}

__device__ double3 operator-=(double3 &lhs, const double3 &rhs) {
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
    return lhs;
}

__device__ double3 operator-=(double3 &lhs, const double &rhs) {
    lhs.x -= rhs;
    lhs.y -= rhs;
    lhs.z -= rhs;
    return lhs;
}

//////////////////////////////////////////
// * operator overloads
//////////////////////////////////////////
__device__ double3 operator*(const double3 &lhs, const double3 &rhs) {
    return make_double3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z);
}

__device__ double3 operator*(const double3 &lhs, const double &rhs) {
    return make_double3(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
}

__device__ double3 operator*(const double &lhs, const double3 &rhs) {
    return make_double3(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z);
}

__device__ double3 operator*=(double3 &lhs, const double3 &rhs) {
    lhs.x *= rhs.x;
    lhs.y *= rhs.y;
    lhs.z *= rhs.z;
    return lhs;
}

__device__ double3 operator*=(double3 &lhs, const double &rhs) {
    lhs.x *= rhs;
    lhs.y *= rhs;
    lhs.z *= rhs;
    return lhs;
}

//////////////////////////////////////////
// / operator overloads
//////////////////////////////////////////

__device__ double3 operator/(const double3 &lhs, const double3 &rhs) {
    return make_double3(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z);
}

__device__ double3 operator/(const double3 &lhs, const double &rhs) {
    return make_double3(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs);
}

__device__ double3 operator/(const double &lhs, const double3 &rhs) {
    return make_double3(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z);
}

__device__ double3 operator/=(double3 &lhs, const double3 &rhs) {
    lhs.x /= rhs.x;
    lhs.y /= rhs.y;
    lhs.z /= rhs.z;
    return lhs;
}

__device__ double3 operator/=(double3 &lhs, const double &rhs) {
    lhs.x /= rhs;
    lhs.y /= rhs;
    lhs.z /= rhs;
    return lhs;
}


//////////////////////////////////////////
// dot product
//////////////////////////////////////////

__device__ double dot(const double3 &lhs, const double3 &rhs) {
    return (lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z);
}

//////////////////////////////////////////
// length
//////////////////////////////////////////

__device__ double length(const double3 &lhs) {
    return sqrt(lhs.x * lhs.x + lhs.y * lhs.y + lhs.z * lhs.z);
}

//////////////////////////////////////////
// normalize
//////////////////////////////////////////

__device__ double3 normalize(const double3 &lhs) {
    double invLen = 1.0 / sqrt(lhs.x * lhs.x + lhs.y * lhs.y + lhs.z * lhs.z);
    return make_double3(lhs.x * invLen, lhs.y * invLen, lhs.z * invLen);
}



