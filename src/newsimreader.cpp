// newsimreader.cpp
#include "newsimreader.h"
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <nlohmann/json.hpp>

template<typename T>
void printFirstThreeImpl(const std::string& name, T* ptr, typename std::enable_if<std::is_same<T, double>::value>::type* = nullptr) {
    printf("%s (first 3 elements): %.6f, %.6f, %.6f\n", name.c_str(), ptr[0], ptr[1], ptr[2]);
}

template<typename T>
void printFirstThreeImpl(const std::string& name, T* ptr, typename std::enable_if<std::is_same<T, uint64_t>::value>::type* = nullptr) {
    printf("%s (first 3 elements): %lu, %lu, %lu\n", name.c_str(), ptr[0], ptr[1], ptr[2]);
}

template<typename T>
void printFirstThreeImpl(const std::string& name, T* ptr, typename std::enable_if<std::is_same<T, int32_t>::value>::type* = nullptr) {
    printf("%s (first 3 elements): %d, %d, %d\n", name.c_str(), ptr[0], ptr[1], ptr[2]);
}

NewSimReader::NewSimReader(std::string configFilePath) : configFilePath(std::move(configFilePath)) {}

std::vector<Variable> NewSimReader::parseJson(const std::string& jsonFilePath) {
    std::ifstream file(jsonFilePath);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open JSON file: " + jsonFilePath);
    }

    nlohmann::json jsonData;
    file >> jsonData;

    std::vector<Variable> parsedVariables;
    for (const auto& item : jsonData.items()) {
        Variable var;
        var.name = item.key();
        var.type = item.value()["dataType"].get<std::string>();
        var.size = item.value()["size"].get<std::vector<int>>();
        var.order = item.value()["order"].get<int>();
        parsedVariables.push_back(var);
    }

    std::sort(parsedVariables.begin(), parsedVariables.end(), 
        [](const Variable& a, const Variable& b) { return a.order < b.order; });

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

    double totalReadTime = 0.0;
    double totalMemory = 0.0;

    for (auto& var : variables) {
        auto varReadStart = std::chrono::high_resolution_clock::now();
        int totalSize = std::accumulate(var.size.begin(), var.size.end(), 1, std::multiplies<int>());

        if (var.type == "double") {
            var.data.doublePtr = new double[totalSize];
            file.read(reinterpret_cast<char*>(var.data.doublePtr), totalSize * sizeof(double));
        } else if (var.type == "float") {
            var.data.floatPtr = new float[totalSize];
            file.read(reinterpret_cast<char*>(var.data.floatPtr), totalSize * sizeof(float));
        } else if (var.type == "uint32") {
            var.data.uint32Ptr = new uint32_t[totalSize];
            file.read(reinterpret_cast<char*>(var.data.uint32Ptr), totalSize * sizeof(uint32_t));
        } else if (var.type == "int32") {
            var.data.int32Ptr = new int32_t[totalSize];
            file.read(reinterpret_cast<char*>(var.data.int32Ptr), totalSize * sizeof(int32_t));
        } else if (var.type == "uint64") {
            var.data.uint64Ptr = new uint64_t[totalSize];
            file.read(reinterpret_cast<char*>(var.data.uint64Ptr), totalSize * sizeof(uint64_t));
        } else if (var.type == "int64") {
            var.data.int64Ptr = new int64_t[totalSize];
            file.read(reinterpret_cast<char*>(var.data.int64Ptr), totalSize * sizeof(int64_t));
        } else {
            fprintf(stderr, "Unsupported data type: %s for variable: %s\n", var.type.c_str(), var.name.c_str());
        }

        auto varReadEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> varReadTime = varReadEnd - varReadStart;

        const double sizeInBytes = totalSize * (var.type == "double" ? sizeof(double) : sizeof(uint64_t));
        std::string sizeStr = formatSize(sizeInBytes);

        totalReadTime += varReadTime.count();
        totalMemory += sizeInBytes;

        printf("%-30s%-30.6f%-30s%d", var.name.c_str(), varReadTime.count(), sizeStr.c_str(), var.size[0]);
        for (size_t i = 1; i < var.size.size(); i++) {
            printf("x%d", var.size[i]);
        }
        printf("%-30s\n", "");
        printf("-----------------------------------------------------------------------------------------------------\n");
    }

    printf("=====================================================================================================\n");
    printf("%-30s%-30.6f%-30s\n", "Total", totalReadTime, formatSize(totalMemory).c_str());
    printf("=====================================================================================================\n");
}

void NewSimReader::extractData(const std::vector<Variable>& variables, uint64_t& particleNum, uint64_t& stepNum, double& stepSize, double& permProb,
                             double& initIn, double& D0, double& d, double& scale, double& tstep, double& vsize,
                             double*& swcmat, uint64_t*& LUT, uint64_t*& C, uint64_t*& pairs, uint64_t*& boundSize,
                             bool debug) {
    printf("Extracting Data\n");
    for (const auto& var : variables) {
        if (var.name == "particle_num" && var.type == "uint64") {
            particleNum = *var.data.uint64Ptr;
        } else if (var.name == "step_num" && var.type == "uint64") {
            stepNum = *var.data.uint64Ptr;
        } else if (var.name == "step_size" && var.type == "double") {
            stepSize = *var.data.doublePtr;
        } else if (var.name == "perm_prob" && var.type == "double") {
            permProb = *var.data.doublePtr;
        } else if (var.name == "init_in" && var.type == "double") {
            initIn = *var.data.doublePtr;
        } else if (var.name == "D0" && var.type == "double") {
            D0 = *var.data.doublePtr;
        } else if (var.name == "dx" && var.type == "double") {
            d = *var.data.doublePtr;
        } else if (var.name == "scale" && var.type == "double") {
            scale = *var.data.doublePtr;
        } else if (var.name == "tstep" && var.type == "double") {
            tstep = *var.data.doublePtr;
        } else if (var.name == "vsize" && var.type == "double") {
            vsize = *var.data.doublePtr;
        } else if (var.name == "swcmat" && var.type == "double") {
            swcmat = var.data.doublePtr;
        } else if (var.name == "LUT" && var.type == "uint64") {
            LUT = var.data.uint64Ptr;
        } else if (var.name == "C" && var.type == "uint64") {
            C = var.data.uint64Ptr;
        } else if (var.name == "pairs" && var.type == "uint64") {
            pairs = var.data.uint64Ptr;
        } else if (var.name == "boundSize" && var.type == "uint64") {
            boundSize = var.data.uint64Ptr;
        }
        else {
            printf("unknown???:::%s", var.name.c_str());
        }
    }
        printf("Done Extracting\n");
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
    const std::vector<Variable>& variables, uint64_t& particleNum, uint64_t& stepNum, double& stepSize, double& permProb,
    double& initIn, double& D0, double& d, double& scale, double& tstep, double& vsize,
    double*& swcmat, uint64_t*& LUT, uint64_t*& C, uint64_t*& pairs, uint64_t*& boundSize,
    bool debug) {

    printf("Extracted Data:\n");

    auto printValue = [](const std::string& name, double value) {
        printf("%s: %.6f\n", name.c_str(), value);
    };
    auto printuint64t = [](const std::string& name, uint64_t value) {
        printf("%s: %llu\n", name.c_str(), value);
    };

    auto printFirstThree = [](const std::string& name, auto* ptr) {
        if (ptr == nullptr) {
            printf("%s: nullptr\n", name.c_str());
        } else {
            printFirstThreeImpl(name, ptr);
        }
    };

    printuint64t("particleNum", particleNum);
    printuint64t("stepNum", stepNum);
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
                printf("%f %f %f %f %f %f\n", 
                    swcmat[i + nrow * 0],
                    swcmat[i + nrow * 1],
                    swcmat[i + nrow * 2],
                    swcmat[i + nrow * 3],
                    swcmat[i + nrow * 4],
                    swcmat[i + nrow * 5]);
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