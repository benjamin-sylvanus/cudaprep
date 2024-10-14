//
// Created by Benjamin Sylvanus on 6/5/23.
//

#include "overloads.h"
#include "cuda_replacements.h"

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
// > operator overloads
//////////////////////////////////////////

int3 operator>(const double3 &lhs, const double3 &rhs) {
    return make_int3(lhs.x > rhs.x, lhs.y > rhs.y, lhs.z > rhs.z);
}


//////////////////////////////////////////
// < operator overloads
//////////////////////////////////////////

int3 operator<(const double3 &lhs, const double3 &rhs) {
    return make_int3(lhs.x < rhs.x, lhs.y < rhs.y, lhs.z < rhs.z);
}

//////////////////////////////////////////
// dot product
//////////////////////////////////////////

double dot(const double3 &lhs, const double3 &rhs) {
    return (lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z);
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
// distance: double3
//////////////////////////////////////////
double distance2(const double3 &lhs, const double3 &rhs) {
    return (lhs.x - rhs.x) * (lhs.x - rhs.x) +
           (lhs.y - rhs.y) * (lhs.y - rhs.y) +
           (lhs.z - rhs.z) * (lhs.z - rhs.z);
}

//////////////////////////////////////////
// distance: double4
//////////////////////////////////////////
double distance2(const double4 &lhs, const double4 &rhs) {
    return (lhs.x - rhs.x) * (lhs.x - rhs.x) +
           (lhs.y - rhs.y) * (lhs.y - rhs.y) +
           (lhs.z - rhs.z) * (lhs.z - rhs.z);
}


//////////////////////////////////////////
// set(double * lhs, int3 &index, double3 &rhs)
//////////////////////////////////////////
void set(double * lhs, int3 &index, double3 &rhs) {
    lhs[index.x] = rhs.x;
    lhs[index.y] = rhs.y;
    lhs[index.z] = rhs.z;
}