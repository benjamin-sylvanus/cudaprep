// newsimreader.h
#pragma once
#include <string>
#include <vector>
#include "Variable.h"

class NewSimReader {
public:
    /**
     * Constructor takes a config file path
     * @param configFilePath Input Path
     */
    explicit NewSimReader(std::string configFilePath);

    /**
     * Parses JSON configuration file
     * @param jsonFilePath Path to JSON file
     * @return Vector of Variable objects
     */
    static std::vector<Variable> parseJson(const std::string& jsonFilePath);

    /**
     * Reads binary file and populates Variable objects
     * @param binaryFilePath Path to binary file
     * @param variables Vector of Variables to populate
     */
    static void readBinaryFile(const std::string& binaryFilePath, std::vector<Variable>& variables);

    /**
     * Extracts data from Variables into respective parameters
     * @param variables Source Variables
     * @param particleNum Number of particles
     * @param stepNum Number of steps
     * @param stepSize Step size
     * @param permProb Permeability probability
     * @param initIn Initial in value
     * @param D0 D0 value
     * @param d d value
     * @param scale Scale value
     * @param tstep Time step
     * @param vsize Volume size
     * @param swcmat SWC matrix
     * @param LUT Lookup table
     * @param C C values
     * @param pairs Pairs values
     * @param boundSize Boundary size
     * @param debug Debug flag
     */
    static void extractData(const std::vector<Variable>& variables, uint64_t& particleNum, uint64_t& stepNum, double& stepSize, double& permProb,
                          double& initIn, double& D0, double& d, double& scale, double& tstep, double& vsize,
                          double*& swcmat, uint64_t*& LUT, uint64_t*& C, uint64_t*& pairs, uint64_t*& boundSize,
                          bool debug = false);

    /**
     * Previews configuration data
     * Same parameters as extractData
     */
    static void previewConfig(const std::vector<Variable>& variables, uint64_t& particleNum, uint64_t& stepNum, double& stepSize, double& permProb,
                            double& initIn, double& D0, double& d, double& scale, double& tstep, double& vsize,
                            double*& swcmat, uint64_t*& LUT, uint64_t*& C, uint64_t*& pairs, uint64_t*& boundSize,
                            bool debug = false);

private:
    std::string configFilePath;
    /**
     * Formats byte size to human-readable string
     * @param bytes Number of bytes
     * @return Formatted string
     */
    static std::string formatSize(double bytes);
};
