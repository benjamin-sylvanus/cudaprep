// NewSimReader.cpp
#include "newsimreader.h"

#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <nlohmann/json.hpp>


NewSimReader::NewSimReader(std::string  configFilePath) : configFilePath(std::move(configFilePath)) {}

std::vector<Variable> NewSimReader::parseJson(const std::string& jsonFilePath) {
    std::ifstream file(jsonFilePath);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open JSON file: " + jsonFilePath);
    }

    nlohmann::json jsonData;
    file >> jsonData;

    std::vector<Variable> parsedVariables;
    for (const auto& [key, value] : jsonData.items()) {
        Variable var;
        var.name = key;
        var.type = value["dataType"].get<std::string>();
        var.size = value["size"].get<std::vector<int>>();
        var.order = value["order"].get<int>();
        parsedVariables.push_back(var);
    }

    std::sort(parsedVariables.begin(), parsedVariables.end(), [](const Variable& a, const Variable& b) {
        return a.order < b.order;
    });

    return parsedVariables;
}

void NewSimReader::readBinaryFile(const std::string& binaryFilePath, std::vector<Variable>& variables) {
    std::ifstream file(binaryFilePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open binary file: " + binaryFilePath);
    }

    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    printf("Binary file size: %.2f MB\n", static_cast<double>(fileSize) / (1024 * 1024));

    printf("=======================================================================\n");
    printf("%-30s%-30s%-30s\n", "Variable Name", "Read Time (ms)", "Size");
    printf("=======================================================================\n");

    for (auto& var : variables) {
        auto varReadStart = std::chrono::high_resolution_clock::now();

        int totalSize = std::accumulate(var.size.begin(), var.size.end(), 1, std::multiplies<int>());

        if (var.type == "double") {
            double* values = new double[totalSize];
            file.read(reinterpret_cast<char*>(values), totalSize * sizeof(double));
            var.data = values;
        } else if (var.type == "float") {
            float* values = new float[totalSize];
            file.read(reinterpret_cast<char*>(values), totalSize * sizeof(float));
            var.data = values;
        } else if (var.type == "uint32") {
            uint32_t* values = new uint32_t[totalSize];
            file.read(reinterpret_cast<char*>(values), totalSize * sizeof(uint32_t));
            var.data = values;
        } else if (var.type == "int32") {
            int32_t* values = new int32_t[totalSize];
            file.read(reinterpret_cast<char*>(values), totalSize * sizeof(int32_t));
            var.data = values;
        } else if (var.type == "uint64") {
            uint64_t* values = new uint64_t[totalSize];
            file.read(reinterpret_cast<char*>(values), totalSize * sizeof(uint64_t));
            var.data = values;
        } else if (var.type == "int64") {
            int64_t* values = new int64_t[totalSize];
            file.read(reinterpret_cast<char*>(values), totalSize * sizeof(int64_t));
            var.data = values;
        } else {
            fprintf(stderr, "Unsupported data type: %s for variable: %s\n", var.type.c_str(), var.name.c_str());
        }

        auto varReadEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> varReadTime = varReadEnd - varReadStart;

        const double sizeInBytes = totalSize * (var.type == "double" ? sizeof(double) : sizeof(uint64_t));
        std::string sizeStr = formatSize(sizeInBytes);

        printf("%-30s%-30.6f%-30s\n", var.name.c_str(), varReadTime.count(), sizeStr.c_str());
        printf("-----------------------------------------------------------------------\n");
    }
}

void NewSimReader::extractData(const std::vector<Variable>& variables, double& particleNum, double& stepNum, double& stepSize, double& permProb,
                               double& initIn, double& D0, double& d, double& scale, double& tstep, double& vsize,
                               double*& swcmat, uint32_t*& LUT, int32_t*& C, uint64_t*& pairs, int64_t*& boundSize,
                               float*& encodings, double*& cfgmat, bool debug) {
    for (const auto& var : variables) {
        if (var.name == "particle_num" && std::holds_alternative<double*>(var.data)) {
            particleNum = *std::get<double*>(var.data);
        } else if (var.name == "step_num" && std::holds_alternative<double*>(var.data)) {
            stepNum = *std::get<double*>(var.data);
        } else if (var.name == "step_size" && std::holds_alternative<float*>(var.data)) {
            stepSize = *std::get<float*>(var.data);
        } else if (var.name == "perm_prob" && std::holds_alternative<float*>(var.data)) {
            permProb = *std::get<float*>(var.data);
        } else if (var.name == "init_in" && std::holds_alternative<double*>(var.data)) {
            initIn = *std::get<double*>(var.data);
        } else if (var.name == "D0" && std::holds_alternative<double*>(var.data)) {
            D0 = *std::get<double*>(var.data);
        } else if (var.name == "d" && std::holds_alternative<double*>(var.data)) {
            d = *std::get<double*>(var.data);
        } else if (var.name == "scale" && std::holds_alternative<double*>(var.data)) {
            scale = *std::get<double*>(var.data);
        } else if (var.name == "tstep" && std::holds_alternative<double*>(var.data)) {
            tstep = *std::get<double*>(var.data);
        } else if (var.name == "vsize" && std::holds_alternative<double*>(var.data)) {
            vsize = *std::get<double*>(var.data);
        } else if (var.name == "swcmat" && std::holds_alternative<double*>(var.data)) {
            swcmat = std::get<double*>(var.data);
        } else if (var.name == "LUT" && std::holds_alternative<uint32_t*>(var.data)) {
            LUT = std::get<uint32_t*>(var.data);
        } else if (var.name == "C" && std::holds_alternative<int32_t*>(var.data)) {
            C = std::get<int32_t*>(var.data);
        } else if (var.name == "pairs" && std::holds_alternative<uint64_t*>(var.data)) {
            pairs = std::get<uint64_t*>(var.data);
        } else if (var.name == "boundSize" && std::holds_alternative<int64_t*>(var.data)) {
            boundSize = std::get<int64_t*>(var.data);
        } else if (var.name == "encodings" && std::holds_alternative<float*>(var.data)) {
            encodings = std::get<float*>(var.data);
        } else if (var.name == "cfgmat" && std::holds_alternative<double*>(var.data)) {
            cfgmat = std::get<double*>(var.data);
        }
    }
}

/**
 *
 * @param bytes
 * @return string
 */
std::string NewSimReader::formatSize(double bytes) {
    const char *units[] = {"B", "KB", "MB", "GB", "TB"};
    int i = 0;
    while (bytes >= 1024 && i < 4) {
        bytes /= 1024;
        i++;
    }
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%.2f %s", bytes, units[i]);
    return std::string(buffer);
}
void NewSimReader::previewConfig(
    const std::vector<Variable>& variables, double& particleNum, double& stepNum, double& stepSize, double& permProb,
    double& initIn, double& D0, double& d, double& scale, double& tstep, double& vsize,
    double*& swcmat, uint32_t*& LUT, int32_t*& C, uint64_t*& pairs, int64_t*& boundSize,
    float*& encodings, double*& cfgmat, bool debug) {

    printf("Extracted Data:\n");

    auto printValue = [](const std::string& name, double value) {
        printf("%s: %.6f\n", name.c_str(), value);
    };

    auto printFirstThree = [](const std::string& name, auto* ptr) {
        if (ptr == nullptr) {
            printf("%s: nullptr\n", name.c_str());
        } else {
            if constexpr (std::is_same_v<std::remove_pointer_t<decltype(ptr)>, double>) {
                printf("%s (first 3 elements): %.6f, %.6f, %.6f\n", name.c_str(), ptr[0], ptr[1], ptr[2]);
            } else if constexpr (std::is_same_v<std::remove_pointer_t<decltype(ptr)>, unsigned long long>) {
                printf("%s (first 3 elements): %llu, %llu, %llu\n", name.c_str(), ptr[0], ptr[1], ptr[2]);
            } else if constexpr (std::is_same_v<std::remove_pointer_t<decltype(ptr)>, int32_t>) {
                printf("%s (first 3 elements): %d, %d, %d\n", name.c_str(), ptr[0], ptr[1], ptr[2]);
            }
        }
    };

    printValue("particleNum", particleNum);
    printValue("stepNum", stepNum);
    printValue("stepSize", stepSize);
    printValue("permProb", permProb);
    printValue("initIn", initIn);
    printValue("D0", D0);
    printValue("d", d);
    printValue("scale", scale);
    printValue("tstep", tstep);
    printValue("vsize", vsize);

    printFirstThree("swcmat", swcmat);
    printFirstThree("LUT", LUT);
    printFirstThree("C", C);
    printFirstThree("pairs", pairs);
    printFirstThree("boundSize", boundSize);
    printFirstThree("encodings", encodings);
    printFirstThree("cfgmat", cfgmat);

    if (debug) {
        printf("Debug: All variables extracted successfully.\n");
    }
}