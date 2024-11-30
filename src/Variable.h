// Variable.h
#pragma once
#include <string>
#include <vector>
#include <cstdint>

class Variable {
public:
    std::string name;
    std::string type;
    std::vector<int> size;
    int order;
    
    union DataPtr {
        double* doublePtr;
        float* floatPtr;
        uint32_t* uint32Ptr;
        int32_t* int32Ptr;
        uint64_t* uint64Ptr;
        int64_t* int64Ptr;
        
        DataPtr() : doublePtr(nullptr) {}
    } data;

    Variable() : order(0) {
        data.doublePtr = nullptr;
    }
};
