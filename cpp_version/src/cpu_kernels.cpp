#include "cpu_kernels.h"
#include "cuda_replacements.h"
#include "funcs.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include "overloads.h"
#include <thread>
#include <mutex>
#include <chrono>

// Constants
const double PI = 3.14159265358979323846;

void volfrac_cpu(const std::vector<double4>& d4swc, const std::vector<int>& nlut, const std::vector<int>& NewIndex,
                  int3 Bounds, int3 IndexSize, int n, std::vector<int>& label, double& vf, bool debug) {
    if (debug) std::cout << "Debug: Entering volfrac_cpu function with " << n << " samples" << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    label.resize(n);
    int R = 0;

    for (int gid = 0; gid < n; ++gid) {
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
                double dist2 = std::pow(parent.x - child.x, 2) + std::pow(parent.y - child.y, 2) + std::pow(parent.z - child.z, 2);
                bool inside = swc2v(A, child, parent, dist2);

                if (inside) {
                    label[gid] = 1;
                    break;
                }
            }
        }
    }
    if (debug) std::cout << "Debug: Label size: " << label.size() << std::endl;
    R = std::accumulate(label.begin(), label.end(), 0);
    vf = (double)R / (double)n;
    if (debug) std::cout << "Debug: Volume fraction calculated: " << vf << std::endl;
}

void simulate_cpu_thread(int start, int end, std::vector<double>& savedata, std::vector<double>& dx2, std::vector<double>& dx4, 
                         int3 Bounds, const std::vector<double>& SimulationParams, const std::vector<double4>& d4swc, 
                         const std::vector<int>& nlut, const std::vector<int>& NewIndex, int3 IndexSize, int size, int iter, 
                         bool debug, double3 point, int SaveAll, std::vector<double>& Reflections, 
                         std::vector<double>& Uref, std::vector<int>& flip, const std::vector<double>& T2, 
                         std::vector<double>& T, std::vector<double>& Sig0, std::vector<double>& SigRe, 
                         const std::vector<double>& BVec, const std::vector<double>& BVal, const std::vector<double>& TD,
                         std::mutex& sig0_mutex) {
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

    for (int gid = start; gid < end; ++gid) {
        if (debug) std::cout << "Debug: Starting simulation for particle " << gid << std::endl;

        double3 A = initPosition(gid, dx2.data(), Bounds, gen,
                                 SimulationParams.data(), 
                                 d4swc.data(),
                                 nlut.data(),
                                 NewIndex.data(),
                                 IndexSize, size, iter, init_in, debug, point);
        double3 xnot = A;
        int2 parstate = make_int2(1, 1);
        std::vector<double> t(Nc, 0);

        if (debug) std::cout << "Debug: Particle " << gid << " initialized at position (" << A.x << ", " << A.y << ", " << A.z << ")" << std::endl;

        for (int i = 0; i < iter; ++i) {
            if (debug && i % 100 == 0) {
                std::cout << "Debug: Particle " << gid << ", Iteration " << i << std::endl;
            }

            double4 xi = make_double4(dis(gen), dis(gen), dis(gen), dis(gen));
            bool completes = xi.w < perm_prob;
            double3 nextpos;
            computeNext(A, step_size, xi, nextpos, PI);

            int3 upper, lower, floorpos;
            validCoord(nextpos, A, Bounds, upper, lower, floorpos, Reflections.data(), Uref.data(), 
                       gid, i, size, iter, flip.data(), debug);
            floorpos = make_int3((int)nextpos.x, (int)nextpos.y, (int)nextpos.z);
            int test_lutvalue = nlut[s2i(floorpos, Bounds)];
            bool inside = checkConnections(IndexSize, test_lutvalue, nextpos,
                                           NewIndex.data(),
                                           d4swc.data(), fstep);

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
                set(savedata.data(), didx, A);
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
                
                {
                    std::lock_guard<std::mutex> lock(sig0_mutex);
                    Sig0[tidx] += s0;
                }

                for (int j = 0; j < 2; j++) {
                    t[j] = 0;
                }
            }
        }

        if (debug) std::cout << "Debug: Simulation completed for particle " << gid << std::endl;
    }
}

void simulate_cpu(std::vector<double>& savedata, std::vector<double>& dx2, std::vector<double>& dx4, 
                  int3 Bounds, const std::vector<double>& SimulationParams, const std::vector<double4>& d4swc, 
                  const std::vector<int>& nlut, const std::vector<int>& NewIndex, int3 IndexSize, int size, int iter, 
                  bool debug, double3 point, int SaveAll, std::vector<double>& Reflections, 
                  std::vector<double>& Uref, std::vector<int>& flip, const std::vector<double>& T2, 
                  std::vector<double>& T, std::vector<double>& Sig0, std::vector<double>& SigRe, 
                  const std::vector<double>& BVec, const std::vector<double>& BVal, const std::vector<double>& TD,
                  int num_threads) {
    if (debug) std::cout << "Debug: Entering simulate_cpu function with " << num_threads << " threads" << std::endl;

    std::vector<std::thread> threads;
    std::mutex sig0_mutex;

    int particles_per_thread = size / num_threads;
    int remaining_particles = size % num_threads;

    auto start_time = std::chrono::high_resolution_clock::now();

    int start = 0;
    for (int i = 0; i < num_threads; ++i) {
        int end = start + particles_per_thread + (i < remaining_particles ? 1 : 0);
        threads.emplace_back(simulate_cpu_thread, start, end, std::ref(savedata), std::ref(dx2), std::ref(dx4),
                             Bounds, std::ref(SimulationParams), std::ref(d4swc), std::ref(nlut), std::ref(NewIndex),
                             IndexSize, size, iter, debug, point, SaveAll, std::ref(Reflections), std::ref(Uref),
                             std::ref(flip), std::ref(T2), std::ref(T), std::ref(Sig0), std::ref(SigRe),
                             std::ref(BVec), std::ref(BVal), std::ref(TD), std::ref(sig0_mutex));
        start = end;
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (debug) {
        std::cout << "Debug: Exiting simulate_cpu function" << std::endl;
        std::cout << "Simulation with " << num_threads << " threads took " << duration.count() / 1000.0 << " seconds" << std::endl;
    }
}
