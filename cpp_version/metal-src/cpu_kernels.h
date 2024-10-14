#pragma once

#include <vector>
#include <cstdint>

// Define float4 struct
struct float4 {
    float x, y, z, w;
    float4 operator+=(const float4& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        w += other.w;
        return *this;
    }
    float4 operator*(float scalar) const {
        return {x * scalar, y * scalar, z * scalar, w * scalar};
    }
};
void processImageOnCPU(std::vector<uint8_t>& imageData, size_t width, size_t height, float brightness);

