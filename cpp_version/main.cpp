#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include "newsimreader.h"
#include "cpu_kernels.h"
#include "cuda_replacements.h"

int main() {
    std::chrono::high_resolution_clock::time_point start_c, stop_c;
    start_c = std::chrono::high_resolution_clock::now();

    // Initialize simulation parameters
    const std::string binaryFile = "/Users/benjaminsylvanus/Documents/mgh/json-writer/simulation_configs/853035674/simulation_config.bin";
    const std::string jsonFile = "/Users/benjaminsylvanus/Documents/mgh/json-writer/simulation_configs/853035674/simulation_config.json";
    
    // Add a debug flag
    bool debug = false;
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
        int size = std::min(static_cast<int>(particleNum), 10000);  // Limit to 10000 particles
        int iter = std::min(static_cast<int>(stepNum), 10000);       // Limit to 10000 iterations
        int SaveAll = 1; // Assuming we want to save all data
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
        std::mt19937 gen(rd());

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

        // Perform simulation
        std::vector<double> SimulationParams = {
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
        debug = true;
        if (debug) {
            std::cout << "Debug: SimulationParams size: " << SimulationParams.size() << std::endl;
            for (size_t i = 0; i < SimulationParams.size(); ++i) {
                std::cout << "SimulationParams[" << i << "]: " << SimulationParams[i] << std::endl;
            }
            std::cout << "Debug: swcmat_vec size: " << swcmat_vec.size() << std::endl;
            std::cout << "Debug: LUT_vec size: " << LUT_vec.size() << std::endl;
            std::cout << "Debug: C_vec size: " << C_vec.size() << std::endl;
            std::cout << "Debug: Calling simulate_cpu..." << std::endl;
        }
        debug = false;

        simulate_cpu(savedata, dx2, dx4, Bounds, SimulationParams,
                     swcmat_vec, LUT_vec, C_vec, 
                     make_int3(C[0], C[1], C[2]), size, iter, debug, 
                     make_double3(swcmat[0], swcmat[1], swcmat[2]), SaveAll, 
                     Reflections, Uref, flip, T2, T, Sig0, SigRe, BVec, BVal, TD);

        if (debug) std::cout << "Debug: simulate_cpu completed." << std::endl;

        stop_c = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_c - start_c);
        std::cout << "Simulation took " << duration.count() / 1000.0 << " seconds" << std::endl;

        debug=true;
        // Debug output after simulation
        if (debug) {
            std::cout << "Debug: Checking results..." << std::endl;
            std::cout << "savedata size: " << savedata.size() << std::endl;
            std::cout << "dx2 size: " << dx2.size() << std::endl;
            std::cout << "dx4 size: " << dx4.size() << std::endl;
            std::cout << "Sig0 size: " << Sig0.size() << std::endl;
            std::cout << "SigRe size: " << SigRe.size() << std::endl;
            // Preview of dx2, Sig0, and SigRe (first 10 non-zero entries)
            std::cout << "dx2: ";
            int count = 0;
            for (size_t i = 0; count < 10 && i < dx2.size(); ++i) {
                if (dx2[i] != 0) {
                    std::cout << dx2[i] << " ";
                    ++count;
                }
            }
            std::cout << std::endl;

            std::cout << "Sig0: ";
            count = 0;
            for (size_t i = 0; count < 10 && i < Sig0.size(); ++i) {
                if (Sig0[i] != 0) {
                    std::cout << Sig0[i] << " ";
                    ++count;
                }
            }
            std::cout << std::endl;

            std::cout << "SigRe: ";
            count = 0;
            for (size_t i = 0; count < 10 && i < SigRe.size(); ++i) {
                if (SigRe[i] != 0) {
                    std::cout << SigRe[i] << " ";
                    ++count;
                }
            }
            std::cout << std::endl;
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
