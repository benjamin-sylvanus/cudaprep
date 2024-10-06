#include <iostream>
#include <string>
#include <vector>
#include "newsimreader.h"
#include "demo_creator.h"
#include <fstream>

int main() {
    DemoCreator demoCreator;
    demoCreator.createDemoFiles("demo_data.bin", "demo_data.json");
    std::cout << "Demo files created successfully." << std::endl;

    const std::string configFile = "default_config.txt";  // Assuming a default config file
    const std::string binaryFile = "demo_data.bin";

    try {
        std::vector<DemoCreator::VariableInfo> variableInfo = demoCreator.getVariableInfo();

        // Create a NewSimReader instance
        NewSimReader simReader(configFile);

        // Parse the JSON file
        std::vector<Variable> variables = NewSimReader::parseJson("demo_data.json");

        // Read the binary file
        NewSimReader::readBinaryFile(binaryFile, variables);
        

        // Initialize variables to hold extracted data
        double particleNum = 0, stepNum = 0, stepSize = 0, permProb = 0;
        double initIn = 0, D0 = 0, d = 0, scale = 0, tstep = 0, vsize = 0;
        double *swcmat = nullptr;
        uint32_t *LUT = nullptr;
        int32_t *C = nullptr;
        uint64_t *pairs = nullptr;
        int64_t *boundSize = nullptr;
        float *encodings = nullptr;
        double *cfgmat = nullptr;

        // Extract data from variables
        simReader.extractData(variables, particleNum, stepNum, stepSize, permProb, initIn, D0, d, scale, tstep, vsize,
                              swcmat, LUT, C, pairs, boundSize, encodings, cfgmat, true);

        std::cout << "Finished reading binary file." << std::endl;
        NewSimReader::previewConfig(variables, particleNum, stepNum, stepSize, permProb, initIn, D0, d, scale, tstep, vsize,
                              swcmat, LUT, C, pairs, boundSize, encodings, cfgmat, true);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}