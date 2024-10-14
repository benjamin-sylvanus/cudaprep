#include "cpu_kernels.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>



void processImageOnCPU(std::vector<uint8_t>& imageData, size_t width, size_t height, float brightness) {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<uint8_t> outputData(width * height * 4);

    float gaussianKernel[5][5] = {
        {1.0/256, 4.0/256, 6.0/256, 4.0/256, 1.0/256},
        {4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256},
        {6.0/256, 24.0/256, 36.0/256, 24.0/256, 6.0/256},
        {4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256},
        {1.0/256, 4.0/256, 6.0/256, 4.0/256, 1.0/256}
    };

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            float4 blurredColor = {0.0f, 0.0f, 0.0f, 0.0f};

            for (int i = -2; i <= 2; ++i) {
                for (int j = -2; j <= 2; ++j) {
                    int nx = x + i;
                    int ny = y + j;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        size_t index = (ny * width + nx) * 4;
                        float4 color = {
                            imageData[index] / 255.0f,
                            imageData[index + 1] / 255.0f,
                            imageData[index + 2] / 255.0f,
                            imageData[index + 3] / 255.0f
                        };
                        blurredColor += color * gaussianKernel[i + 2][j + 2];
                    }
                }
            }

            size_t index = (y * width + x) * 4;
            outputData[index] = std::min(blurredColor.x * brightness, 1.0f) * 255;
            outputData[index + 1] = std::min(blurredColor.y * brightness, 1.0f) * 255;
            outputData[index + 2] = std::min(blurredColor.z * brightness, 1.0f) * 255;
            outputData[index + 3] = blurredColor.w * 255;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "CPU image processing completed in " << elapsed.count() << " ms." << std::endl;
}
