#pragma once
#include <string>
#include <vector>
#include <variant>
#include <cstdint>

struct Variable {
    std::string name;
    std::string type;
    std::vector<int> size;
    int order;
    std::variant<double*, float*, 
                 int8_t*, uint8_t*, 
                 int16_t*, uint16_t*, 
                 int32_t*, uint32_t*, 
                 int64_t*, uint64_t*, 
                 std::string> data;
};