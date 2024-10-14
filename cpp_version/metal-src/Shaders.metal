#include <metal_stdlib>
using namespace metal;

kernel void process_image(texture2d<float, access::read> inTexture [[texture(0)]],
                          texture2d<float, access::write> outTexture [[texture(1)]],
                          constant float &brightness [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height()) {
        return;
    }

    float4 color = inTexture.read(gid);
    float4 adjustedColor = float4(min(color.rgb * brightness, 1.0), color.a);

    // Apply a 5x5 Gaussian blur
    float gaussianKernel[5][5] = {
        {1.0/256, 4.0/256, 6.0/256, 4.0/256, 1.0/256},
        {4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256},
        {6.0/256, 24.0/256, 36.0/256, 24.0/256, 6.0/256},
        {4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256},
        {1.0/256, 4.0/256, 6.0/256, 4.0/256, 1.0/256}
    };

    float4 blurredColor = float4(0.0);
    for (int i = -2; i <= 2; ++i) {
        for (int j = -2; j <= 2; ++j) {
            uint2 offset = uint2(gid.x + i, gid.y + j);
            if (offset.x < inTexture.get_width() && offset.y < inTexture.get_height()) {
                blurredColor += inTexture.read(offset) * gaussianKernel[i+2][j+2];
            }
        }
    }

    // Additional operations can be added here, e.g., edge detection

    outTexture.write(blurredColor, gid);
}
