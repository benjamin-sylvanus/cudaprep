#include "demo_creator.h"
#include "Variable.h"
#include <fstream>
#include <random>
#include <numeric>

DemoCreator::DemoCreator() : particleNum(1000) {}

void DemoCreator::createDemoFiles(const std::string& binaryFilename, const std::string& jsonFilename) {
    std::vector<Variable> variables = generateDemoData();

    // Write binary file
    std::ofstream binaryFile(binaryFilename, std::ios::binary);
    if (!binaryFile) {
        throw std::runtime_error("Unable to open binary file for writing: " + binaryFilename);
    }

    for (const auto& var : variables) {
        int totalSize = std::accumulate(var.size.begin(), var.size.end(), 1, std::multiplies<int>());
        if (var.type == "double") {
            binaryFile.write(reinterpret_cast<const char*>(std::get<double*>(var.data)), totalSize * sizeof(double));
        } else if (var.type == "float") {
            binaryFile.write(reinterpret_cast<const char*>(std::get<float*>(var.data)), totalSize * sizeof(float));
        } else if (var.type == "uint32") {
            binaryFile.write(reinterpret_cast<const char*>(std::get<uint32_t*>(var.data)), totalSize * sizeof(uint32_t));
        } else if (var.type == "int32") {
            binaryFile.write(reinterpret_cast<const char*>(std::get<int32_t*>(var.data)), totalSize * sizeof(int32_t));
        } else if (var.type == "uint64") {
            binaryFile.write(reinterpret_cast<const char*>(std::get<uint64_t*>(var.data)), totalSize * sizeof(uint64_t));
        } else if (var.type == "int64") {
            binaryFile.write(reinterpret_cast<const char*>(std::get<int64_t*>(var.data)), totalSize * sizeof(int64_t));
        }
    }

    binaryFile.close();

    // Write JSON file
    nlohmann::json jsonData = generateJsonData(variables);
    std::ofstream jsonFile(jsonFilename);
    if (!jsonFile) {
        throw std::runtime_error("Unable to open JSON file for writing: " + jsonFilename);
    }

    jsonFile << jsonData.dump(4);
    jsonFile.close();
}

std::vector<Variable> DemoCreator::generateDemoData() {
    std::vector<Variable> variables;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0);
    std::uniform_int_distribution<uint32_t> uint32_dis(0, 100);
    std::uniform_int_distribution<int32_t> int32_dis(-100, 100);
    std::uniform_int_distribution<uint64_t> uint64_dis(0, 100);
    std::uniform_int_distribution<int64_t> int64_dis(-100, 100);

    auto createVar = [&](const std::string& name, const std::string& type, int order, const std::vector<int>& size, auto generator) {
        Variable var;
        var.name = name;
        var.type = type;
        var.size = size;
        var.order = order;

        int totalSize = std::accumulate(size.begin(), size.end(), 1, std::multiplies<int>());
        using T = decltype(generator());
        T* data = new T[totalSize];
        for (int i = 0; i < totalSize; ++i) {
            data[i] = generator();
        }
        var.data = data;
        variables.push_back(var);
    };

    // Generate all variables
    createVar("particle_num", "double", 0, {1, 1}, [&]() { return static_cast<double>(particleNum); });
    createVar("step_num", "double", 1, {1, 1}, [&]() { return 1000.0; });
    createVar("step_size", "float", 2, {1, 1}, [&]() { return static_cast<float>(dis(gen)); });
    createVar("perm_prob", "float", 3, {1,1}, [&]() { return static_cast<float>(dis(gen)); });
    createVar("init_in", "double", 4, {1, 1}, [&]() { return dis(gen); });
    createVar("D0", "double", 5, {1, 1}, [&]() { return dis(gen); });
    createVar("d", "double", 6, {1, 1}, [&]() { return dis(gen); });
    createVar("scale", "double", 7, {1, 1}, [&]() { return dis(gen); });
    createVar("tstep", "double", 8, {1, 1}, [&]() { return dis(gen); });
    createVar("vsize", "double", 9, {1, 1}, [&]() { return dis(gen); });
    createVar("swcmat", "double", 10, {particleNum, 4}, [&]() { return dis(gen); });
    createVar("LUT", "uint32", 11, {particleNum, 1}, [&]() { return uint32_dis(gen); });
    createVar("C", "int32", 12, {particleNum, 1}, [&]() { return int32_dis(gen); });
    createVar("pairs", "uint64", 13, {particleNum, 2}, [&]() { return uint64_dis(gen); });
    createVar("boundSize", "int64", 14, {3, 1}, [&]() { return int64_dis(gen); });
    createVar("encodings", "float", 15, {particleNum, 3}, [&]() { return static_cast<float>(dis(gen)); });
    createVar("cfgmat", "double", 16, {particleNum, 3}, [&]() { return dis(gen); });

    return variables;
}

nlohmann::json DemoCreator::generateJsonData(const std::vector<Variable>& variables) {
    nlohmann::json data;
    
    for (const auto& var : variables) {
        data[var.name] = {
            {"dataType", var.type},
            {"size", var.size},
            {"order", var.order}
        };
    }

    return data;
}

std::vector<DemoCreator::VariableInfo> DemoCreator::getVariableInfo() const {
    return {
        {"particle_num", "double", {1}},
        {"step_num", "double", {1}},
        {"step_size", "double", {1}},
        {"perm_prob", "double", {1}},
        {"init_in", "double", {1}},
        {"D0", "double", {1}},
        {"d", "double", {1}},
        {"scale", "double", {1}},
        {"tstep", "double", {1}},
        {"vsize", "double", {1}},
        {"swcmat", "double", {static_cast<int>(particleNum * 4)}},
        {"LUT", "uint32", {static_cast<int>(particleNum)}},  // Changed from "int64" to "unsigned long long"
        {"C", "int32", {static_cast<int>(particleNum)}},  // Changed from "int64" to "unsigned long long"
        {"pairs", "uint64", {static_cast<int>(particleNum * 2)}},  // Changed from "int64" to "unsigned long long"
        {"boundSize", "unsigned long long", {3}},  // Changed from "int64" to "unsigned long long"
        {"encodings", "double", {static_cast<int>(particleNum * 3)}},
        {"cfgmat", "double", {static_cast<int>(particleNum * 3)}}
    };
}