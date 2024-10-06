#include "cpu_kernels.h"
#include "cuda_replacements.h"
#include "funcs.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>

// Add these function declarations
int s2i(int3 i, int3 b);
bool swc2v(double3 nextpos, double4 child, double4 parent, double dist);
double distance2(const double4 &lhs, const double4 &rhs);
double3 initPosition(int gid, double *dx2, int3 &Bounds, std::mt19937 &gen,
                     double *SimulationParams, double4 *d4swc, int *nlut, int *NewIndex,
                     int3 &IndexSize, int size, int iter, int init_in, bool debug, double3 point);
void computeNext(double3 &A, double &step, double4 &xi, double3 &nextpos, double &pi);
void validCoord(double3 &nextpos, double3 &pos, int3 &b_int3, int3 &upper, int3 &lower, int3 &floorpos,
                double *reflections, double *uref, int gid, int i, int size, int iter, int *flips);
bool checkConnections(int3 i_int3, int test_lutvalue, double3 nextpos, int *NewIndex, double4 *d4swc, double &fstep);

// Constants
const double PI = 3.14159265358979323846;

void volfrac_cpu(const std::vector<double4>& d4swc, const std::vector<int>& nlut, const std::vector<int>& NewIndex,
                  int3 Bounds, int3 IndexSize, int n, std::vector<int>& label, double& vf) {
    // Mark unused parameter
    (void)n;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    int N = 10000000;
    label.resize(N);
    int R = 0;

    for (int gid = 0; gid < N; ++gid) {
        label[gid] = 0;
        double3 A = make_double3(dis(gen) * Bounds.x, dis(gen) * Bounds.y, dis(gen) * Bounds.z);
        int3 floorpos = make_int3((int)A.x, (int)A.y, (int)A.z);
        int id_test = s2i(floorpos, Bounds);
        int test_lutvalue = nlut[id_test];

        for (int page = 0; page < IndexSize.z; ++page) {
            int3 c_new = make_int3(test_lutvalue, 0, page);
            int3 p_new = make_int3(test_lutvalue, 1, page);
            int2 vindex = make_int2(NewIndex[s2i(c_new, IndexSize)] - 1, NewIndex[s2i(p_new, IndexSize)] - 1);

            if (vindex.x != -1) {
                double4 child = d4swc[vindex.x];
                double4 parent = d4swc[vindex.y];
                double dist2 = distance2(parent, child);
                bool inside = swc2v(A, child, parent, dist2);

                if (inside) {
                    label[gid] = 1;
                    break;
                }
            }
        }
    }

    R = std::accumulate(label.begin(), label.end(), 0);
    vf = (double)R / (double)N;
}

void simulate_cpu(std::vector<double>& savedata, std::vector<double>& dx2, std::vector<double>& dx4, 
                  int3 Bounds, const std::vector<double>& SimulationParams, const std::vector<double4>& d4swc, 
                  const std::vector<int>& nlut, const std::vector<int>& NewIndex, int3 IndexSize, int size, int iter, 
                  bool debug, double3 point, int SaveAll, std::vector<double>& Reflections, 
                  std::vector<double>& Uref, std::vector<int>& flip, const std::vector<double>& T2, 
                  std::vector<double>& T, std::vector<double>& Sig0, std::vector<double>& SigRe, 
                  const std::vector<double>& BVec, const std::vector<double>& BVal, const std::vector<double>& TD) {
    // Mark unused parameters
    (void)T;
    (void)SigRe;
    (void)BVec;
    (void)BVal;
    (void)TD;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    double step_size = SimulationParams[2];
    double perm_prob = SimulationParams[3];
    int init_in = (int)SimulationParams[4];
    double tstep = SimulationParams[8];
    double vsize = SimulationParams[9];
    int Tstep = iter / timepoints;
    double fstep = 1;

    for (int gid = 0; gid < size; ++gid) {
        double3 A = initPosition(gid, dx2.data(), Bounds, gen,
                                 const_cast<double*>(SimulationParams.data()), 
                                 const_cast<double4*>(d4swc.data()),
                                 const_cast<int*>(nlut.data()),
                                 const_cast<int*>(NewIndex.data()),
                                 IndexSize, size, iter, init_in, debug, point);
        double3 xnot = A;
        int2 parstate = make_int2(1, 1);
        std::vector<double> t(Nc, 0);

        for (int i = 0; i < iter; ++i) {
            double4 xi = make_double4(dis(gen), dis(gen), dis(gen), dis(gen));
            bool completes = xi.w < perm_prob;
            double3 nextpos;
            double pi = PI;
            computeNext(A, step_size, xi, nextpos, pi);

            int3 upper, lower, floorpos;
            validCoord(nextpos, A, Bounds, upper, lower, floorpos, Reflections.data(), Uref.data(), 
                       gid, i, size, iter, flip.data());
            floorpos = make_int3((int)nextpos.x, (int)nextpos.y, (int)nextpos.z);
            int test_lutvalue = nlut[s2i(floorpos, Bounds)];
            bool inside = checkConnections(IndexSize, test_lutvalue, nextpos,
                                           const_cast<int*>(NewIndex.data()),
                                           const_cast<double4*>(d4swc.data()), fstep);

            parstate.y = inside ? 1 : 0;

            // Update particle position and time in compartments
            if (parstate.x == parstate.y) {
                A = nextpos;
                t[parstate.x] += tstep;
            } else if (parstate.x && !parstate.y) {
                if (completes) {
                    A = nextpos;
                    parstate.x = parstate.y;
                    t[0] += tstep * fstep;
                    t[1] += tstep * (1 - fstep);
                } else {
                    t[0] += tstep;
                }
            } else if (!parstate.x && parstate.y) {
                if (completes) {
                    A = nextpos;
                    parstate.x = parstate.y;
                    t[0] += tstep * (1 - fstep);
                    t[1] += tstep * fstep;
                } else {
                    t[1] += tstep;
                }
            }

            // Store Position Data
            if (SaveAll) {
                int3 dix = make_int3(size, iter, 3);
                int3 did[3] = {make_int3(gid, i, 0), make_int3(gid, i, 1), make_int3(gid, i, 2)};
                int3 didx = make_int3(s2i(did[0], dix), s2i(did[1], dix), s2i(did[2], dix));
                setData(savedata.data(), didx, A);
            }

            // Store Tensor Data
            double3 d2 = make_double3(0.0, 0.0, 0.0);
            diffusionTensor(&A, &xnot, vsize, dx2.data(), dx4.data(), &d2, i, gid, iter, size);

            // Store Signal Data
            if (i % Tstep == 0) {
                int tidx = i / Tstep;
                double s0 = 0.0;
                for (int j = 0; j < 2; j++) {
                    s0 += t[j] / T2[j];
                }
                s0 = std::exp(-s0);
                Sig0[tidx] += s0;

                for (int j = 0; j < 2; j++) {
                    t[j] = 0;
                }
            }
        }
    }
}

// Add these function declarations at the top of the file
void setData(double* data, int3 idx, double3 value);
void diffusionTensor(double3* A, double3* xnot, double vsize, double* dx2, double* dx4, double3* d2, int i, int gid, int iter, int size);

double distance2(const double4 &lhs, const double4 &rhs) {
    double dx = lhs.x - rhs.x;
    double dy = lhs.y - rhs.y;
    double dz = lhs.z - rhs.z;
    return dx*dx + dy*dy + dz*dz;
}

void validCoord(double3 &nextpos, double3 &pos, int3 &b_int3, int3 &upper, int3 &lower, int3 &floorpos,
                double *reflections, double *uref, int gid, int i, int size, int iter, int *flips) {
    // Mark unused parameters
    (void)pos;
    (void)reflections;
    (void)uref;
    (void)gid;
    (void)i;
    (void)size;
    (void)iter;
    (void)flips;

    // Implement the logic for validCoord here
    upper = make_int3(b_int3.x - 1, b_int3.y - 1, b_int3.z - 1);
    lower = make_int3(0, 0, 0);
    floorpos = make_int3((int)nextpos.x, (int)nextpos.y, (int)nextpos.z);

    // Ensure nextpos is within bounds
    nextpos.x = std::max(0.0, std::min((double)upper.x, nextpos.x));
    nextpos.y = std::max(0.0, std::min((double)upper.y, nextpos.y));
    nextpos.z = std::max(0.0, std::min((double)upper.z, nextpos.z));
}

void setData(double* data, int3 idx, double3 value) {
    int index = idx.x * 3 + idx.y * 3 * idx.z + idx.z;
    data[index] = value.x;
    data[index + 1] = value.y;
    data[index + 2] = value.z;
}