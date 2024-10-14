// NewSimReader.h
#pragma once
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "Variable.h"
#include <cstdint>

class NewSimReader {
public:
    /**
     *
     * @param configFilePath Input Path
     */
    explicit NewSimReader(std::string  configFilePath);
    /**
     * 
     * @param jsonFilePath 
     * @return 
     */
    static std::vector<Variable> parseJson(const std::string& jsonFilePath);
    /**
     * 
     * @param binaryFilePath
     * @param variables
     */
    static void readBinaryFile(const std::string& binaryFilePath, std::vector<Variable>& variables);
    /**
     * 
     * @param variables 
     * @param particleNum 
     * @param stepNum 
     * @param stepSize 
     * @param permProb 
     * @param initIn 
     * @param D0 
     * @param d 
     * @param scale 
     * @param tstep 
     * @param vsize 
     * @param swcmat 
     * @param LUT 
     * @param C 
     * @param pairs 
     * @param boundSize 
     * @param debug 
     */
    static void extractData(const std::vector<Variable>& variables, double& particleNum, double& stepNum, double& stepSize, double& permProb,
                     double& initIn, double& D0, double& d, double& scale, double& tstep, double& vsize,
                     double*& swcmat, uint64_t*& LUT, uint64_t*& C, uint64_t*& pairs, uint64_t*& boundSize, bool debug = false);

    /**
     * 
     * @param variables 
     * @param particleNum 
     * @param stepNum 
     * @param stepSize 
     * @param permProb 
     * @param initIn 
     * @param D0 
     * @param d 
     * @param scale 
     * @param tstep 
     * @param vsize 
     * @param swcmat 
     * @param LUT 
     * @param C 
     * @param pairs 
     * @param boundSize 
     * @param debug 
     */
    static void previewConfig(const std::vector<Variable>& variables, double& particleNum, double& stepNum, double& stepSize, double& permProb,
                       double& initIn, double& D0, double& d, double& scale, double& tstep, double& vsize,
                       double*& swcmat, uint64_t*& LUT, uint64_t*& C, uint64_t*& pairs, uint64_t*& boundSize,
                       bool debug = false);

private:
    std::string configFilePath;
    static std::string formatSize(double bytes);
};
