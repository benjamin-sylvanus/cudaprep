#ifndef DEMO_CREATOR_H
#define DEMO_CREATOR_H

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "Variable.h"

class DemoCreator {
public:
    DemoCreator();
    void createDemoFiles(const std::string& binaryFilename, const std::string& jsonFilename);

    struct VariableInfo {
        std::string name;
        std::string type;
        std::vector<int> size;
    };
    std::vector<VariableInfo> getVariableInfo() const;

private:
    std::vector<Variable> generateDemoData();
    nlohmann::json generateJsonData(const std::vector<Variable>& variables);
    int particleNum;  // Add this line
};

#endif // DEMO_CREATOR_H