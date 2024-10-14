#ifndef FUNCS_H
#define FUNCS_H

#include "cuda_replacements.h"
#include <vector>
#include <cstdint>
#include <random>  // Add this line to include the random header

/**
 * @param u_dx2 diffusion tensor
 * @param u_dx4 diffusion tensor
 * @param u_SimP simulation parameters
 * @param u_D4Swc swc data
 * @param u_NewLut lookuptable
 * @param u_NewIndex index array
 * @param u_Flip reflection counter
 * @param simparam simulation parameters
 * @param swc_trim swc data
 * @param lut lookuptable
 * @param indexarr index array
 * @param bounds bounding box
 * @param nrow number of rows
 * @param prod product of dimensions
 * @param newindexsize size of index array
 * @param sa_size size of the saveall array
 * @param Nbvec number of vectors
 * @param timepoints number of timepoints
 * @param NC number of connections
 * @brief Sets up the data for the simulation
 */
void setup_data(double * u_dx2, double * u_dx4, double * u_SimP, double4 * u_D4Swc, int * u_NewLut,
                                    int * u_NewIndex, int * u_Flip, double * simparam, double4 * swc_trim,
                                    double * mdx2, double * mdx4, double * u_AllData,double * u_Reflections, 
                                    double * u_Uref, double * u_T2, double * u_T, double * u_SigRe, double * u_Sig0,
                                    double * u_bvec, double * u_bval, double * u_TD, std::vector<uint64_t> lut, std::vector<uint64_t> indexarr, 
                                    std::vector<uint64_t> bounds, int size, int iter, int nrow, int prod, int newindexsize, int sa_size, int Nbvec, int timepoints, int NC, int n_vf, double * u_vf, int * u_label);

int s2i(int3 i, int3 b);
bool swc2v(double3 nextpos, double4 child, double4 parent, double dist);
double distance2(const double4 &lhs, const double4 &rhs);
double3 initPosition(int gid, const double *dx2, int3 &Bounds, std::mt19937 &gen,
                     const double *SimulationParams, const double4 *d4swc, const int *nlut, const int *NewIndex,
                     int3 &IndexSize, int size, int iter, int init_in, bool debug, double3 point);

void computeNext(double3 &A, double step, double4 &xi, double3 &nextpos, const double &pi);

void validCoord(double3 &nextpos, double3 &pos, int3 &b_int3, int3 &upper, int3 &lower, int3 &floorpos,
                double *reflections, double *uref, int gid, int i, int size, int iter, int *flips, bool debug);
bool checkConnections(int3 i_int3, int test_lutvalue, double3 nextpos, const int *NewIndex, const double4 *d4swc, double &fstep); 

void diffusionTensor(double3 *A, double3 *xnot, double vsize, double *dx2, double *dx4, double3 *d2, int i, int gid, int iter, int size);
double dot(double3 a, double3 b);
#endif // FUNCS_H