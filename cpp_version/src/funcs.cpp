#include "funcs.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <sys/stat.h>
#include "cuda_replacements.h"
#include "overloads.h"

bool swc2v(double3 nextpos, double4 child, double4 parent, double dist) {
    bool pos;
    double cr2, pr2;
    if (dist == 0) {
        cr2 = std::pow(child.w, 2);
        pr2 = std::pow(parent.w, 2);
        pos = (std::pow(nextpos.x - child.x, 2) + std::pow(nextpos.y - child.y, 2) + std::pow(nextpos.z - child.z, 2)) < cr2 ||
              (std::pow(nextpos.x - parent.x, 2) + std::pow(nextpos.y - parent.y, 2) + std::pow(nextpos.z - parent.z, 2)) < pr2;
    } else {
        double t = ((nextpos.x - child.x) * (parent.x - child.x) + (nextpos.y - child.y) * (parent.y - child.y) +
                    (nextpos.z - child.z) * (parent.z - child.z)) / dist;
        double x = child.x + (parent.x - child.x) * t;
        double y = child.y + (parent.y - child.y) * t;
        double z = child.z + (parent.z - child.z) * t;
        cr2 = std::pow(child.w, 2);
        pr2 = std::pow(parent.w, 2);
        double L = std::sqrt(std::pow(parent.x-child.x,2) + std::pow(parent.y-child.y,2) + std::pow(parent.z-child.z,2));
        double sin_theta = std::fabs(child.w - parent.w)/L;
        double cos_theta = std::sqrt(1-std::pow(sin_theta,2));
        bool list1 = (t > ( (child.w * sin_theta) / L)) && (t < (1 + (parent.w * sin_theta) / L));
        if (list1) {
            double dist2 = (std::pow(nextpos.x - x, 2) + std::pow(nextpos.y - y, 2) + std::pow(nextpos.z - z, 2));
            double m = (parent.w-child.w) * cos_theta/(L+(parent.w-child.w)*sin_theta);
            double rx = L * t;
            double ry = m * rx + (cos_theta - m * sin_theta) * child.w;
            pos = dist2 < std::pow(ry,2);
        } else {
            pos = (std::pow(nextpos.x - child.x, 2) + std::pow(nextpos.y - child.y, 2) + std::pow(nextpos.z - child.z, 2)) < cr2 ||
                  (std::pow(nextpos.x - parent.x, 2) + std::pow(nextpos.y - parent.y, 2) + std::pow(nextpos.z - parent.z, 2)) < pr2;
        }
    }
    return pos;
}

int s2i(int3 i, int3 b) {
    return b.x * (b.y * i.z + i.y) + i.x;
}

double3 initPosition(int gid, const double *dx2, int3 &Bounds, std::mt19937 &gen,
                     const double *SimulationParams, const double4 *d4swc, const int *nlut, const int *NewIndex,
                     int3 &IndexSize, int size, int iter, int init_in, bool debug, double3 point) {
    std::uniform_real_distribution<> dis(0.0, 1.0);
    double3 nextpos, A;
    int3 upper, lower, floorpos;
    int3 b_int3 = make_int3(Bounds.x, Bounds.y, Bounds.z);
    int3 i_int3 = make_int3(IndexSize.x, IndexSize.y, IndexSize.z);
    double3 b_d3 = make_double3((double)b_int3.x, (double)b_int3.y, (double)b_int3.z);
    double4 xr;
    int2 parstate;
    parstate.y = 0;
    bool cont = true;

    if (init_in == 1) {
        while (cont) {
            xr = make_double4(dis(gen), dis(gen), dis(gen), dis(gen));
            A = make_double3(xr.x * b_d3.x, xr.y * b_d3.y, xr.z * b_d3.z);
            nextpos = A;
            floorpos = make_int3((int)A.x, (int)A.y, (int)A.z);
            upper = make_int3(floorpos.x < b_int3.x, floorpos.y < b_int3.y, floorpos.z < b_int3.z);
            lower = make_int3(floorpos.x > 0, floorpos.y > 0, floorpos.z > 0);
            parstate.x = (lower.x && lower.y && lower.z && upper.x && upper.y && upper.z) ? 1 : 0;
            double4 parent, child;
            int2 vindex;
            int id_test = s2i(floorpos, b_int3);
            int test_lutvalue = nlut[id_test];
            double dist2;
            if (parstate.x) {
                for (int page = 0; page < i_int3.z; page++) {
                    int3 c_new = make_int3(test_lutvalue, 0, page);
                    int3 p_new = make_int3(test_lutvalue, 1, page);
                    vindex.x = NewIndex[s2i(c_new, i_int3)] - 1;
                    vindex.y = NewIndex[s2i(p_new, i_int3)] - 1;
                    if ((vindex.x) != -1) {
                        child = d4swc[vindex.x];
                        parent = d4swc[vindex.y];
                        dist2 = std::pow(parent.x - child.x, 2) + std::pow(parent.y - child.y, 2) + std::pow(parent.z - child.z, 2);
                        bool inside = swc2v(nextpos, child, parent, dist2);
                        if (inside) {
                            parstate.y = 1;
                            cont = false;
                            break;
                        }
                    }
                }
            }
        }
    } else if (init_in == 2) {
        xr = make_double4(dis(gen), dis(gen), dis(gen), dis(gen));
        A = make_double3(xr.x * b_d3.x, xr.y * b_d3.y, xr.z * b_d3.z);
    } else if (init_in == 3) {
        A = point;
    }
    return A;
}

void diffusionTensor(double3 *A, double3 *xnot, double vsize, double *dx2, double *dx4, double3 *d2, int i, int gid, int iter, int size) {
    // if (gid == 1) {
    //     printf("A: %f %f %f\n", A->x, A->y, A->z);
    //     printf("xnot: %f %f %f\n", xnot->x, xnot->y, xnot->z);
    //     printf("vsize: %f\n", vsize);
    //     printf("dx2: %f %f %f %f %f %f\n", dx2[0], dx2[1], dx2[2], dx2[3], dx2[4], dx2[5]);
    //     printf("dx4: %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", dx4[0], dx4[1], dx4[2], dx4[3], dx4[4], dx4[5], dx4[6], dx4[7], dx4[8], dx4[9], dx4[10], dx4[11], dx4[12], dx4[13], dx4[14]);
    //     printf("d2: %f %f %f\n", d2->x, d2->y, d2->z);
    //     printf("i: %d\n", i);
    //     printf("gid: %d\n", gid);
    //     printf("iter: %d\n", iter);
    //     printf("size: %d\n", size);
    // }
    d2->x = std::fabs((A->x - xnot->x) * vsize);
    d2->y = std::fabs((A->y - xnot->y) * vsize);
    d2->z = std::fabs((A->z - xnot->z) * vsize);

    dx2[6 * i + 0] += d2->x * d2->x;
    dx2[6 * i + 1] += d2->x * d2->y;
    dx2[6 * i + 2] += d2->x * d2->z;
    dx2[6 * i + 3] += d2->y * d2->y;
    dx2[6 * i + 4] += d2->y * d2->z;
    dx2[6 * i + 5] += d2->z * d2->z;

    dx4[15 * i + 0] += d2->x * d2->x * d2->x * d2->x;
    dx4[15 * i + 1] += d2->x * d2->x * d2->x * d2->y;
    dx4[15 * i + 2] += d2->x * d2->x * d2->x * d2->z;
    dx4[15 * i + 3] += d2->x * d2->x * d2->y * d2->y;
    dx4[15 * i + 4] += d2->x * d2->x * d2->y * d2->z;
    dx4[15 * i + 5] += d2->x * d2->x * d2->z * d2->z;
    dx4[15 * i + 6] += d2->x * d2->y * d2->y * d2->y;
    dx4[15 * i + 7] += d2->x * d2->y * d2->y * d2->z;
    dx4[15 * i + 8] += d2->x * d2->y * d2->z * d2->z;
    dx4[15 * i + 9] += d2->x * d2->z * d2->z * d2->z;
    dx4[15 * i + 10] += d2->y * d2->y * d2->y * d2->y;
    dx4[15 * i + 11] += d2->y * d2->y * d2->y * d2->z;
    dx4[15 * i + 12] += d2->y * d2->y * d2->z * d2->z;
    dx4[15 * i + 13] += d2->y * d2->z * d2->z * d2->z;
    dx4[15 * i + 14] += d2->z * d2->z * d2->z * d2->z;
}

double3 setNextPos(double3 nextpos, double3 A, double4 xi, double step) {
    nextpos.x = A.x + ((2.0 * xi.x - 1.0) * step);
    nextpos.y = A.y + ((2.0 * xi.y - 1.0) * step);
    nextpos.z = A.z + ((2.0 * xi.z - 1.0) * step);
    return nextpos;
}

void writeResults(double *w_swc, double *hostSimP, double *hostdx2, double *mdx2, double *hostdx4,
                  double *mdx4, double *u_t, double *u_Reflections, double *u_uref, double *u_Sig0,
                  double *u_SigRe, double *u_AllData, int iter, int size, int nrow, int timepoints, int Nbvec, int sa_size, int SaveAll,
                  std::string outpath) {
    struct stat sb;
    std::cout << "Outpath: " << outpath << std::endl;
    if (stat(outpath.c_str(), &sb) == 0)
        std::cout << "Valid Path" << std::endl;
    else if (mkdir(outpath.c_str(), 0777) == -1)
        std::cerr << "Error: " << std::strerror(errno) << std::endl;
    else
        std::cout << "Directory created" << std::endl;

    std::vector<double> t(iter);
    double tstep = hostSimP[8];
    for (int i = 0; i < iter; i++) {
        t[i] = tstep * i;
        for (int j = 0; j < 6; j++)
            mdx2[6 * i + j] = (hostdx2[6 * i + j] / size) / (2.0 * t[i]);
        for (int j = 0; j < 15; j++)
            mdx4[15 * i + j] = (hostdx4[15 * i + j] / size) / (2.0 * t[i]);
    }

    auto write_binary = [](const std::string &path, const double *data, size_t size) {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char *>(data), size * sizeof(double));
    };

    write_binary(outpath + "/swc.bin", w_swc, nrow * 4);
    write_binary(outpath + "/outcfg.bin", hostSimP, 10);
    write_binary(outpath + "/dx2.bin", hostdx2, 6 * iter);
    write_binary(outpath + "/mdx2.bin", mdx2, 6 * iter);
    write_binary(outpath + "/dx4.bin", hostdx4, 15 * iter);
    write_binary(outpath + "/mdx4.bin", mdx4, 15 * iter);
    write_binary(outpath + "/t.bin", t.data(), iter);
    write_binary(outpath + "/reflections.bin", u_Reflections, iter * size * 3);
    write_binary(outpath + "/uref.bin", u_uref, iter * size * 3);
    write_binary(outpath + "/sig0.bin", u_Sig0, timepoints);
    write_binary(outpath + "/sigRe.bin", u_SigRe, Nbvec * iter);

    if (SaveAll) {
        write_binary(outpath + "/allData.bin", u_AllData, 3 * sa_size);
    }
}

void computeNext(double3 &A, double step, double4 &xi, double3 &nextpos, const double &pi) {
    double theta = 2 * pi * xi.x;
    double v = xi.y;
    double cos_phi = 2 * v - 1;
    double sin_phi = std::sqrt(1 - std::pow(cos_phi, 2));
    nextpos.x = A.x + (step * sin_phi * std::cos(theta));
    nextpos.y = A.y + (step * sin_phi * std::sin(theta));
    nextpos.z = A.z + (step * cos_phi);
}

bool checkConnections(int3 i_int3, int test_lutvalue, double3 nextpos, const int *NewIndex, const double4 *d4swc, double &fstep) {
    int3 vindex;
    double4 child, parent;
    double dist2;

    for (int page = 0; page < i_int3.z; page++) {
        int3 c_new = make_int3(test_lutvalue, 0, page);
        int3 p_new = make_int3(test_lutvalue, 1, page);
        vindex.x = NewIndex[s2i(c_new, i_int3)] - 1;
        vindex.y = NewIndex[s2i(p_new, i_int3)] - 1;

        if ((vindex.x) != -1) {
            child = d4swc[vindex.x];
            parent = d4swc[vindex.y];
            dist2 = std::pow(parent.x - child.x, 2) + std::pow(parent.y - child.y, 2) + std::pow(parent.z - child.z, 2);
            bool inside = swc2v(nextpos, child, parent, dist2);
            if (inside) {
                return true;
            }
        } else {
            return false;
        }
    }
    return false;
}


void setup_data(double * u_dx2, double * u_dx4, double * u_SimP, double4 * u_D4Swc, int * u_NewLut,
                                    int * u_NewIndex, int * u_Flip, double * simparam, double4 * swc_trim,
                                    double * mdx2, double * mdx4, double * u_AllData,double * u_Reflections, 
                                    double * u_Uref, double * u_T2, double * u_T, double * u_SigRe, double * u_Sig0,
                                    double * u_bvec, double * u_bval, double * u_TD, std::vector<uint64_t> lut, std::vector<uint64_t> indexarr, 
                                    std::vector<uint64_t> bounds, int size, int iter, int nrow, int prod, int newindexsize, int sa_size, int Nbvec, int timepoints, int NC, int n_vf, double * u_vf, int * u_label) {
    // Set Values for Host
    {
        memset(u_dx2, 0.0, 6 * iter * sizeof(double));
        memset(u_dx4, 0.0, 15 * iter * sizeof(double));

        {
            for (int i = 0; i < 10; i++) {
                u_SimP[i] = simparam[i];
            }
            for (int i = 0; i < nrow; i++) {
                u_D4Swc[i].x = swc_trim[i].x;
                u_D4Swc[i].y = swc_trim[i].y;
                u_D4Swc[i].z = swc_trim[i].z;
                u_D4Swc[i].w = swc_trim[i].w;
            }
            for (int i = 0; i < prod; i++) {
                u_NewLut[i] = lut[i];
            }
            for (int i = 0; i < newindexsize; i++) {
                u_NewIndex[i] = indexarr[i];
            }
        }

        memset(mdx2, 0.0, 6 * iter * sizeof(double));
        memset(mdx4, 0.0, 15 * iter * sizeof(double));
        memset(u_AllData, 0.0, 3 * sa_size * sizeof(double));
        memset(u_Reflections, 0.0, 3 * iter * size * sizeof(double));
        memset(u_Uref, 0.0, 3 * iter * size * sizeof(double));
        memset(u_Flip, 0.0, 3 * iter * sizeof(int));
        memset(u_T2, 0.0, NC * sizeof(double));                            // T2 is read from file?
        memset(u_T, 0.0, NC * sizeof(double));                             // T is set to 0.0
        memset(u_SigRe, 0.0, Nbvec * timepoints * sizeof(double));         // Calculated in kernel
        memset(u_Sig0, 0.0, timepoints * sizeof(double));                  // Calculated in kernel
        memset(u_bvec, 0.0, Nbvec * 3 * sizeof(double));                   // bvec is read from file
        memset(u_bval, 0.0, Nbvec * sizeof(double));                       // bval is read from file
        memset(u_TD, 0.0,   Nbvec * sizeof(double));                       // TD is read from file

        memset(u_vf, 0.0, sizeof(double));                                 // Calc in vf kernel
        memset(u_label, 0.0,  n_vf * sizeof(int));                         // arg in or default
        printf("Set Host Values\n");
    }
}



void validCoord(double3 &nextpos, double3 &pos, int3 &b_int3, int3 &upper, int3 &lower, int3 &floorpos,
                double *reflections, double *uref, int gid, int i, int size, int iter, int *flips, bool debug) {
    double3 High = make_double3((double)b_int3.x, (double)b_int3.y, (double)b_int3.z);
    double3 Low = make_double3(0.0, 0.0, 0.0);

    int3 dix = make_int3(size, iter, 3);
    int3 did[4];
    did[0] = make_int3(gid, i, 0);
    did[1] = make_int3(gid, i, 1);
    did[2] = make_int3(gid, i, 2);
    did[3] = make_int3(s2i(did[0], dix), s2i(did[1], dix), s2i(did[2], dix));

    int count = 0;
    while(true) {
        int3 UPPER = make_int3(nextpos.x > High.x, nextpos.y > High.y, nextpos.z > High.z);
        int3 LOWER = make_int3(nextpos.x < Low.x, nextpos.y < Low.y, nextpos.z < Low.z);

        if (!(UPPER.x || UPPER.y || UPPER.z || LOWER.x || LOWER.y || LOWER.z)) {
            return; // no reflection needed
        }

        double3 normal;
        double3 pointOnPlane;
        int fidx;

        if (LOWER.x) {
            fidx = 6*gid + 0;
            pointOnPlane = make_double3(Low.x, nextpos.y, nextpos.z);
            normal = make_double3(1.0, 0.0, 0.0);
        } else if (UPPER.x) {
            fidx = 6*gid + 1;
            pointOnPlane = make_double3(High.x, nextpos.y, nextpos.z);
            normal = make_double3(-1.0, 0.0, 0.0);
        } else if (LOWER.y) {
            fidx = 6*gid + 2;
            pointOnPlane = make_double3(nextpos.x, Low.y, nextpos.z);
            normal = make_double3(0.0, 1.0, 0.0);
        } else if (UPPER.y) {
            fidx = 6*gid + 3;
            pointOnPlane = make_double3(nextpos.x, High.y, nextpos.z);
            normal = make_double3(0.0, -1.0, 0.0);
        } else if (LOWER.z) {
            fidx = 6*gid + 4;
            pointOnPlane = make_double3(nextpos.x, nextpos.y, Low.z);
            normal = make_double3(0.0, 0.0, 1.0);
        } else if (UPPER.z) {
            fidx = 6*gid + 5;
            pointOnPlane = make_double3(nextpos.x, nextpos.y, High.z);
            normal = make_double3(0.0, 0.0, -1.0);
        }

        double D = -(dot(normal, pointOnPlane));
        double3 d = pos - nextpos;
        double t1 = -((dot(normal, nextpos) + D)) / dot(normal, d);
        double3 intersectionPoint = nextpos + d * t1;
        double3 reflectionVector = nextpos - intersectionPoint;
        reflectionVector = reflectionVector - normal * (2 * dot(reflectionVector, normal));

        double3 unreflected = nextpos;
        nextpos = intersectionPoint + reflectionVector;

        if (debug) {
            printf("NextPos: %f %f %f -> %f %f %f\n", unreflected.x, unreflected.y, unreflected.z, 
                   nextpos.x, nextpos.y, nextpos.z);
            printf("Count: %d\n", count);
        }
        count += 1;

        set(reflections, did[3], intersectionPoint);
        set(uref, did[3], unreflected);

        nextpos = intersectionPoint + reflectionVector;
        flips[fidx] += 1;
    }
}

double dot(double3 a, double3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
