#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <thread>
#include "newsimreader.h"
#include "cpu_kernels.h"
#include "cuda_replacements.h"
#include "funcs.h"

int main() {
    std::chrono::high_resolution_clock::time_point start_c, stop_c;
    start_c = std::chrono::high_resolution_clock::now();

    // Initialize simulation parameters
    const std::string binaryFile = "/Users/benjaminsylvanus/Documents/mgh/json-writer/simulation_configs/853035674/simulation_config.bin";
    const std::string jsonFile = "/Users/benjaminsylvanus/Documents/mgh/json-writer/simulation_configs/853035674/simulation_config.json";
    
    // Add a debug flag
    bool debug;
    debug = false;
    try {

        // Create a NewSimReader instance
        NewSimReader simReader("configFile");

        // Parse the JSON file
        std::vector<Variable> variables = NewSimReader::parseJson(jsonFile);

        // Read the binary file
        NewSimReader::readBinaryFile(binaryFile, variables);

        // Initialize variables to hold extracted data
        double particleNum = 0, stepNum = 0, stepSize = 0, permProb = 0;
        double initIn = 0, D0 = 0, d = 0, scale = 0, tstep = 0, vsize = 0;
        double *swcmat = nullptr;
        uint64_t *LUT = nullptr;
        uint64_t *C = nullptr;
        uint64_t *pairs = nullptr;
        uint64_t *boundSize = nullptr;

        // Extract data from variables
        NewSimReader::extractData(variables, particleNum, stepNum, stepSize, permProb, initIn, D0, d, scale, tstep,
                                  vsize,
                                  swcmat, LUT, C, pairs, boundSize, true);
        NewSimReader::previewConfig(variables, particleNum, stepNum, stepSize, permProb, initIn, D0, d, scale, tstep, vsize,
                                  swcmat, LUT, C, pairs, boundSize, true);
        // Set up simulation parameters
        int size =  50000;
        int iter =  10000;
        int SaveAll = 0; // Assuming we want to save all data
        int3 Bounds = make_int3(boundSize[0], boundSize[1], boundSize[2]);
        
        // Allocate memory for simulation data
        std::vector<double> savedata(3 * size * iter);
        std::vector<double> dx2(6 * iter);
        std::vector<double> dx4(15 * iter);
        std::vector<double> Reflections(3 * iter * size);
        std::vector<double> Uref(3 * iter * size);
        std::vector<int> flip(3 * size);
        std::vector<double> T2(Nc);
        std::vector<double> T(Nc);
        std::vector<double> Sig0(timepoints);
        std::vector<double> SigRe(Nbvec * timepoints);
        std::vector<double> BVec(Nbvec * 3);
        std::vector<double> BVal(Nbvec);
        std::vector<double> TD(Nbvec);

        // Set up random number generator
        std::random_device rd;

        // Determine nrow from swcmat
        size_t nrow = 0;
        for (const auto&[name, type, size, order, data] : variables) {
            if (name == "swcmat") {
                if (size.size() >= 2) {
                    nrow = size[0];  // The first dimension of swcmat
                    std::cout << "Number of rows in swcmat: " << nrow << std::endl;
                } else {
                    std::cerr << "Error: swcmat does not have the expected dimensions" << std::endl;
                    return 1;
                }
                break;
            }
        }

        if (nrow == 0) {
            std::cerr << "Error: Could not determine the number of rows in swcmat" << std::endl;
            return 1;
        }

        // Calculate prod (total number of voxels)
        size_t prod = boundSize[0] * boundSize[1] * boundSize[2];

        // Perform volume fraction calculation
        int volfrac_samples = 10000;  // Or any other desired number of samples
        std::vector<int> label(volfrac_samples);
        double vf = 0.0;
        std::vector swcmat_vec(reinterpret_cast<double4*>(swcmat), reinterpret_cast<double4*>(swcmat) + nrow);
        std::vector<int> LUT_vec(LUT, LUT + prod);
        std::vector<int> C_vec(C, C + 3);
        volfrac_cpu(swcmat_vec, LUT_vec, C_vec, Bounds, 
                    make_int3(C[0], C[1], C[2]), volfrac_samples, label, vf, debug);

        std::cout << "Volume Fraction: " << vf << std::endl;

        // Debug output
        if (debug) {
            std::cout << "Debug: Preparing for simulation..." << std::endl;
            std::cout << "Size: " << size << ", Iter: " << iter << std::endl;
            std::cout << "Bounds: " << Bounds.x << ", " << Bounds.y << ", " << Bounds.z << std::endl;
        }

        // After extracting data from variables
        std::vector SimulationParams = {
            static_cast<double>(size),  // particle_num
            static_cast<double>(iter),  // step_num
            stepSize,
            permProb,
            initIn,
            D0,
            d,
            scale,
            tstep,
            vsize
        };

        // Debug output
        if (debug) {
            std::cout << "Debug: SimulationParams:" << std::endl;
            for (size_t i = 0; i < SimulationParams.size(); ++i) {
                std::cout << "  [" << i << "]: " << SimulationParams[i] << std::endl;
            }
        }
        // Reset other data structures as needed
        unsigned int num_threads = std::thread::hardware_concurrency() / 4;

        // Perform simulation with 4 thread
        std::cout << "Running simulation with " << num_threads << " threads..." << std::endl;
        start_c = std::chrono::high_resolution_clock::now();
        simulate_cpu(savedata, dx2, dx4, Bounds, SimulationParams,
                     swcmat_vec, LUT_vec, C_vec, 
                     make_int3(C[0], C[1], C[2]), size, iter, debug, 
                     make_double3(swcmat[0], swcmat[1], swcmat[2]), SaveAll, 
                     Reflections, Uref, flip, T2, T, Sig0, SigRe, BVec, BVal, TD, num_threads);

        stop_c = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_c - start_c);
        std::cout << "Simulation took " << duration.count() / 1000.0 << " seconds" << std::endl;
        debug=true;
        if (debug) {
            logSimulationResults(savedata, dx2, dx4, Sig0, SigRe);
        }
        debug=false;
        // Reset Sig0 and other relevant data structures
        std::fill(Sig0.begin(), Sig0.end(), 0.0);
        std::fill(dx2.begin(), dx2.end(), 0.0);
        std::fill(dx4.begin(), dx4.end(), 0.0);
        std::fill(Reflections.begin(), Reflections.end(), 0.0);
        std::fill(Uref.begin(), Uref.end(), 0.0);
        std::fill(flip.begin(), flip.end(), 0);
        std::fill(T2.begin(), T2.end(), 0.0);
        std::fill(T.begin(), T.end(), 0.0);
        std::fill(SigRe.begin(), SigRe.end(), 0.0);
        std::fill(BVec.begin(), BVec.end(), 0.0);
        std::fill(BVal.begin(), BVal.end(), 0.0);
        std::fill(TD.begin(), TD.end(), 0.0);
        // Reset other data structures as needed
        num_threads = std::max(static_cast<int>(std::thread::hardware_concurrency() / 2), 12);
        // Perform simulation with 10 threads
        std::cout << "Running simulation with " << num_threads << " threads..." << std::endl;
        start_c = std::chrono::high_resolution_clock::now();
        simulate_cpu(savedata, dx2, dx4, Bounds, SimulationParams,
                     swcmat_vec, LUT_vec, C_vec, 
                     make_int3(C[0], C[1], C[2]), size, iter, debug, 
                     make_double3(swcmat[0], swcmat[1], swcmat[2]), SaveAll, 
                     Reflections, Uref, flip, T2, T, Sig0, SigRe, BVec, BVal, TD, num_threads);

        stop_c = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_c - start_c);
        std::cout << "Simulation took " << duration.count() / 1000.0 << " seconds" << std::endl;

        debug=true;
        // Debug output after simulation
        if (debug) {
            logSimulationResults(savedata, dx2, dx4, Sig0, SigRe);
        }



        // Write results (you'll need to implement this function)
        // writeResults(swcmat, SimulationParams, dx2, dx4, T, Reflections, Uref, Sig0, SigRe, savedata, 
        //              iter, size, swcmat.size() / 4, timepoints, Nbvec, size * iter, SaveAll, "output_path");

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
