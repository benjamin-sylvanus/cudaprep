// NewSimReader.cpp
#include "NewSimReader.h"
#include <fstream>
#include <iostream>
#include <numeric>

NewSimReader::NewSimReader(const std::string& configFilePath)
    : configFilePath(configFilePath) {}

void NewSimReader::readJsonFile(const std::string& jsonFilePath) {
    std::ifstream file(jsonFilePath);
    nlohmann::json jsonData;
    file >> jsonData;
    for (const auto& item : jsonData.items()) {
        variables.push_back(createVariableFromJsonItem(item.key(), item.value()));
    }
}

void NewSimReader::readBinaryFile(const std::string& binaryFilePath) {
    this->binaryFilePath = binaryFilePath;
    std::ifstream file(binaryFilePath, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Failed to open binary file: " << binaryFilePath << std::endl;
        return;
    }

    std::sort(variables.begin(), variables.end(), [](const Variable& a, const Variable& b) {
        return a.order < b.order;
    });

    for (auto& var : variables) {
        int totalSize = std::accumulate(var.size.begin(), var.size.end(), 1, std::multiplies<int>());

        if (var.type == "double") {
            double* values = new double[totalSize];
            file.read(reinterpret_cast<char*>(values), totalSize * sizeof(double));
            var.data = values;
        }
        else if (var.type == "uint64" || var.type == "int64") {
            int64_t* values = new int64_t[totalSize];
            file.read(reinterpret_cast<char*>(values), totalSize * sizeof(int64_t));
            var.data = values;
        }
        else if (var.type == "string") {
            std::string value(totalSize, '\0');
            file.read(&value[0], totalSize);
            var.data = std::move(value);
        }
    }
}

std::vector<std::vector<uint64_t>> NewSimReader::readDims(const std::string& dimsFilePath) {
    std::vector<std::vector<uint64_t>> dims;
    std::ifstream file(dimsFilePath);
    std::string line;
    while (std::getline(file, line)) {
        std::vector<uint64_t> dimVec;
        std::stringstream ss(line);
        uint64_t dim;
        while (ss >> dim) {
            dimVec.push_back(dim);
        }
        dims.push_back(dimVec);
    }
    return dims;
}


template<typename T>
std::vector<T> NewSimReader::read(const std::string& binFilePath) {
    std::vector<T> data;
    std::ifstream file(binFilePath, std::ios::binary);
    if (file.is_open()) {
        file.seekg(0, std::ios::end);
        size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        size_t numElements = fileSize / sizeof(T);
        data.resize(numElements);
        file.read(reinterpret_cast<char*>(data.data()), fileSize);
        file.close();
    }
    return data;
}


void NewSimReader::extractData(double& particleNum, double& stepNum, double& stepSize, double& permProb,
                               double& initIn, double& D0, double& d, double& scale, double& tstep, double& vsize,
                               double* swcmat, int64_t* LUT, int64_t* C, int64_t* pairs, int64_t* boundSize,
                               double* encodings, double* cfgmat) {
    for (const auto& var : variables) {
        if (var.name == "particle_num") {
            particleNum = *std::get<double*>(var.data);
        }
        else if (var.name == "step_num") {
            stepNum = *std::get<double*>(var.data);
        }
        else if (var.name == "step_size") {
            stepSize = *std::get<double*>(var.data);
        }
        else if (var.name == "perm_prob") {
            permProb = *std::get<double*>(var.data);
        }
        else if (var.name == "init_in") {
            initIn = *std::get<double*>(var.data);
        }
        else if (var.name == "D0") {
            D0 = *std::get<double*>(var.data);
        }
        else if (var.name == "d") {
            d = *std::get<double*>(var.data);
        }
        else if (var.name == "scale") {
            scale = *std::get<double*>(var.data);
        }
        else if (var.name == "tstep") {
            tstep = *std::get<double*>(var.data);
        }
        else if (var.name == "vsize") {
            vsize = *std::get<double*>(var.data);
        }
        else if (var.name == "swcmat") {
            int swcmatSize = std::accumulate(var.size.begin(), var.size.end(), 1, std::multiplies<int>());
            std::memcpy(swcmat, std::get<double*>(var.data), swcmatSize * sizeof(double));
        }
        else if (var.name == "LUT") {
            int LUTSize = std::accumulate(var.size.begin(), var.size.end(), 1, std::multiplies<int>());
            std::memcpy(LUT, std::get<int64_t*>(var.data), LUTSize * sizeof(int64_t));
        }
        else if (var.name == "C") {
            int CSize = std::accumulate(var.size.begin(), var.size.end(), 1, std::multiplies<int>());
            std::memcpy(C, std::get<int64_t*>(var.data), CSize * sizeof(int64_t));
        }
        else if (var.name == "pairs") {
            int pairsSize = std::accumulate(var.size.begin(), var.size.end(), 1, std::multiplies<int>());
            std::memcpy(pairs, std::get<int64_t*>(var.data), pairsSize * sizeof(int64_t));
        }
        else if (var.name == "boundSize") {
            int boundSizeSize = std::accumulate(var.size.begin(), var.size.end(), 1, std::multiplies<int>());
            std::memcpy(boundSize, std::get<int64_t*>(var.data), boundSizeSize * sizeof(int64_t));
        }
        else if (var.name == "encodings") {
            int encodingsSize = std::accumulate(var.size.begin(), var.size.end(), 1, std::multiplies<int>());
            std::memcpy(encodings, std::get<double*>(var.data), encodingsSize * sizeof(double));
        }
        else if (var.name == "cfgmat") {
            int cfgmatSize = std::accumulate(var.size.begin(), var.size.end(), 1, std::multiplies<int>());
            std::memcpy(cfgmat, std::get<double*>(var.data), cfgmatSize * sizeof(double));
        }
    }
}
