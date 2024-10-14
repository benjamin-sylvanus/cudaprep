//
// Created by Benjamin Sylvanus on 6/5/23.
//

#ifndef OVERLOADS_H
#define OVERLOADS_H

#include "cuda_replacements.h"

// Replace __device__ with  
double3 operator+(const double3 &lhs, const double3 &rhs);
double3 operator+(const double3 &lhs, const double &rhs);
double3 operator+(const double &lhs, const double3 &rhs);
double3 operator+=(double3 &lhs, const double3 &rhs);
double3 operator+=(double3 &lhs, const double &rhs);
double3 operator*(const double3 &lhs, const double3 &rhs);
double3 operator*(const double3 &lhs, const double &rhs);
double3 operator*(const double &lhs, const double3 &rhs);
double3 operator*=(double3 &lhs, const double3 &rhs);
double3 operator-(const double3 &lhs, const double3 &rhs);
double3 operator-(const double3 &lhs, const double &rhs);
double3 operator-(const double &lhs, const double3 &rhs);
double3 operator-=(double3 &lhs, const double3 &rhs);
double3 operator-=(double3 &lhs, const double &rhs);




void set(double * lhs, int3 &index, double3 &rhs);
// Add other operator overloads as needed

#endif // OVERLOADS_H
