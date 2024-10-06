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
    static std::vector<Variable> parseJson(const std::string& jsonFilePath);
    static void readBinaryFile(const std::string& binaryFilePath, std::vector<Variable>& variables);
    static void extractData(const std::vector<Variable>& variables, double& particleNum, double& stepNum, double& stepSize, double& permProb,
                     double& initIn, double& D0, double& d, double& scale, double& tstep, double& vsize,
                     double*& swcmat, uint32_t*& LUT, int32_t*& C, uint64_t*& pairs, int64_t*& boundSize,
                     float*& encodings, double*& cfgmat, bool debug = false);

    static void previewConfig(const std::vector<Variable>& variables, double& particleNum, double& stepNum, double& stepSize, double& permProb,
                       double& initIn, double& D0, double& d, double& scale, double& tstep, double& vsize,
                       double*& swcmat, uint32_t*& LUT, int32_t*& C, uint64_t*& pairs, int64_t*& boundSize,
                       float*& encodings, double*& cfgmat, bool debug = false);

private:
    std::string configFilePath;
    static std::string formatSize(double bytes);
};
