// NewSimReader.h
#pragma once
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "Variable.h"

class NewSimReader {
public:
    NewSimReader(const std::string& configFilePath);
    void readJsonFile(const std::string& jsonFilePath);
    void readBinaryFile(const std::string& binaryFilePath);
    std::vector<std::vector<uint64_t>> readDims(const std::string& dimsFilePath);
    template<typename T>
    std::vector<T> read(const std::string& binFilePath);
    void extractData(double& particleNum, double& stepNum, double& stepSize, double& permProb,
                     double& initIn, double& D0, double& d, double& scale, double& tstep, double& vsize,
                     double* swcmat, int64_t* LUT, int64_t* C, int64_t* pairs, int64_t* boundSize,
                     double* encodings, double* cfgmat);

private:
    std::string configFilePath;
    std::string binaryFilePath;
    std::vector<Variable> variables;
};
