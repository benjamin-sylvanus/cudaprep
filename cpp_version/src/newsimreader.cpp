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
    const std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    printf("Binary file size: %.2f MB\n", static_cast<double>(fileSize) / (1024 * 1024));
    printf("=====================================================================================================\n");
    printf("%-30s%-30s%-30s%-30s\n", "Variable Name", "Read Time (ms)", "Size", "Dimensions");
    printf("=====================================================================================================\n");
    for (auto&[name, type, size, order, data] : variables) {
        auto varReadStart = std::chrono::high_resolution_clock::now();
        int totalSize = std::accumulate(size.begin(), size.end(), 1, std::multiplies<int>());

        if (type == "double") {
            auto* values = new double[totalSize];
            file.read(reinterpret_cast<char*>(values), totalSize * sizeof(double));
            data = values;
        } else if (type == "float") {
            auto* values = new float[totalSize];
            file.read(reinterpret_cast<char*>(values), totalSize * sizeof(float));
            data = values;
        } else if (type == "uint32") {
            auto values = new uint32_t[totalSize];
            file.read(reinterpret_cast<char*>(values), totalSize * sizeof(uint32_t));
            data = values;
        } else if (type == "int32") {
            auto* values = new int32_t[totalSize];
            file.read(reinterpret_cast<char*>(values), totalSize * sizeof(int32_t));
            data = values;
        } else if (type == "uint64") {
            auto* values = new uint64_t[totalSize];
            file.read(reinterpret_cast<char*>(values), totalSize * sizeof(uint64_t));
            data = values;
        } else if (type == "int64") {
            auto* values = new int64_t[totalSize];
            file.read(reinterpret_cast<char*>(values), totalSize * sizeof(int64_t));
            data = values;
        } else {
            fprintf(stderr, "Unsupported data type: %s for variable: %s\n", type.c_str(), name.c_str());
        }

        auto varReadEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> varReadTime = varReadEnd - varReadStart;

        const double sizeInBytes = totalSize * (type == "double" ? sizeof(double) : sizeof(uint64_t));
        std::string sizeStr = formatSize(sizeInBytes);

        printf("%-30s%-30.6f%-30s%d", name.c_str(), varReadTime.count(), sizeStr.c_str(), size[0]);
        for (int i=1; i<size.size(); i++) {
            printf("x%d", size[i]);
        }
        printf("%-30s\n","");
        printf("-----------------------------------------------------------------------------------------------------\n");
    }
}

void NewSimReader::extractData(const std::vector<Variable>& variables, double& particleNum, double& stepNum, double& stepSize, double& permProb,
                               double& initIn, double& D0, double& d, double& scale, double& tstep, double& vsize,
                               double*& swcmat, uint64_t*& LUT, uint64_t*& C, uint64_t*& pairs, uint64_t*& boundSize,
                               bool debug) {
    for (const auto& var : variables) {
        if (var.name == "particle_num" && std::holds_alternative<double*>(var.data)) {
            particleNum = *std::get<double*>(var.data);
        } else if (var.name == "step_num" && std::holds_alternative<double*>(var.data)) {
            stepNum = *std::get<double*>(var.data);
        } else if (var.name == "step_size" && std::holds_alternative<double*>(var.data)) {
            stepSize = *std::get<double*>(var.data);
        } else if (var.name == "perm_prob" && std::holds_alternative<double*>(var.data)) {
            permProb = *std::get<double*>(var.data);
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
        } else if (var.name == "LUT" && std::holds_alternative<uint64_t*>(var.data)) {
            LUT = std::get<uint64_t*>(var.data);
        } else if (var.name == "C" && std::holds_alternative<uint64_t*>(var.data)) {
            C = std::get<uint64_t*>(var.data);
        } else if (var.name == "pairs" && std::holds_alternative<uint64_t*>(var.data)) {
            pairs = std::get<uint64_t*>(var.data);
        } else if (var.name == "boundSize" && std::holds_alternative<uint64_t*>(var.data)) {
            boundSize = std::get<uint64_t*>(var.data);
        }
    }
}


std::string NewSimReader::formatSize(double bytes) {
    const char *units[] = {"B", "KB", "MB", "GB", "TB"};
    int i = 0;
    while (bytes >= 1024 && i < 4) {
        bytes /= 1024;
        i++;
    }
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%.2f %s", bytes, units[i]);
    return {buffer};
}

void NewSimReader::previewConfig(
    const std::vector<Variable>& variables, double& particleNum, double& stepNum, double& stepSize, double& permProb,
    double& initIn, double& D0, double& d, double& scale, double& tstep, double& vsize,
    double*& swcmat, uint64_t*& LUT, uint64_t*& C, uint64_t*& pairs, uint64_t*& boundSize,
    bool debug) {

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

    printf("particleNum = %.0f\n", particleNum);
    printValue("stepNum", stepNum);
    printValue("stepSize", stepSize);
    printValue("permProb", permProb);
    printValue("initIn", initIn);
    printValue("D0", D0);
    printValue("d", d);
    printValue("scale", scale);
    printValue("tstep", tstep);
    printValue("vsize", vsize);


    for (const auto& var : variables) {
        if (var.name == "swcmat") {
            printf("size of swcmat = %dx%d\n", var.size[0], var.size[1]);
            const int nrow = var.size[0];
            for (int i = 0; i < 20; i++) {
                printf("%f %f %f %f %f %f\n", swcmat[i + nrow * 0],swcmat[i + nrow * 1],swcmat[i + nrow * 2],swcmat[i + nrow * 3],swcmat[i + nrow * 4],swcmat[i + nrow * 5]);
            }
        }
    }
    printFirstThree("LUT", LUT);
    printFirstThree("C", C);
    printFirstThree("pairs", pairs);
    printFirstThree("boundSize", boundSize);

    if (debug) {
        printf("Debug: All variables extracted successfully.\n");
    }
}
