#import <MetalKit/MetalKit.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include "cpu_kernels.h"

@interface MetalCompute : NSObject

@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, strong) id<MTLComputePipelineState> pipelineState;

- (instancetype)initWithDevice:(id<MTLDevice>)device;
- (void)processImageWithBrightness:(float)brightness;
- (NSUInteger)width;
- (NSUInteger)height;
- (void)getInputTextureData:(std::vector<uint8_t>&)imageData;

@end

@implementation MetalCompute {
    id<MTLTexture> _inputTexture;
    id<MTLTexture> _outputTexture;
    MTLTextureDescriptor *_textureDescriptor;
    NSUInteger _width;
    NSUInteger _height;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device {
    self = [super init];
    if (self) {
        _device = device;
        _commandQueue = [_device newCommandQueue];
        _width = 16384;  // 8K width
        _height = 16384; // 8K height
        [self loadMetal];
        [self setupTextures];
    }
    return self;
}

- (NSUInteger)width {
    return _width;
}

- (NSUInteger)height {
    return _height;
}

- (void)loadMetal {
    NSError *error = nil;
    
    // Load the shader source code
    NSString *shaderPath = [[NSBundle mainBundle] pathForResource:@"Shaders" ofType:@"metal"];
    if (!shaderPath) {
        NSLog(@"Failed to find shader file.");
        return;
    }
    
    std::ifstream shaderFile([shaderPath UTF8String]);
    std::stringstream shaderStream;
    shaderStream << shaderFile.rdbuf();
    std::string shaderSource = shaderStream.str();
    NSString *shaderSourceString = [NSString stringWithUTF8String:shaderSource.c_str()];
    
    // Compile the shader
    id<MTLLibrary> library = [_device newLibraryWithSource:shaderSourceString options:nil error:&error];
    if (!library) {
        NSLog(@"Failed to create shader library: %@", error);
        return;
    }
    
    id<MTLFunction> kernelFunction = [library newFunctionWithName:@"process_image"];
    if (!kernelFunction) {
        NSLog(@"Failed to find the kernel function.");
        return;
    }
    
    _pipelineState = [_device newComputePipelineStateWithFunction:kernelFunction error:&error];
    if (!_pipelineState) {
        NSLog(@"Failed to create pipeline state object: %@", error);
    }
}

- (void)setupTextures {
    _textureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                            width:_width
                                                                           height:_height
                                                                        mipmapped:NO];
    _textureDescriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    
    _inputTexture = [_device newTextureWithDescriptor:_textureDescriptor];
    _outputTexture = [_device newTextureWithDescriptor:_textureDescriptor];
    
    [self generateInputTexture];
}

- (void)generateInputTexture {
    uint8_t *imageData = (uint8_t*)malloc(_width * _height * 4);
    for (int y = 0; y < _height; y++) {
        for (int x = 0; x < _width; x++) {
            int index = (y * _width + x) * 4;
            imageData[index] = (x + y) % 256;     // R
            imageData[index + 1] = x % 256;       // G
            imageData[index + 2] = y % 256;       // B
            imageData[index + 3] = 255;           // A
        }
    }
    
    MTLRegion region = MTLRegionMake2D(0, 0, _width, _height);
    [_inputTexture replaceRegion:region mipmapLevel:0 withBytes:imageData bytesPerRow:_width * 4];
    
    free(imageData);
}

- (void)processImageWithBrightness:(float)brightness {
    auto start = std::chrono::high_resolution_clock::now();

    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    
    [computeEncoder setComputePipelineState:_pipelineState];
    [computeEncoder setTexture:_inputTexture atIndex:0];
    [computeEncoder setTexture:_outputTexture atIndex:1];
    [computeEncoder setBytes:&brightness length:sizeof(float) atIndex:0];
    
    MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
    MTLSize gridSize = MTLSizeMake(_width, _height, 1);
    
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [computeEncoder endEncoding];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    
    std::cout << "GPU image processing completed in " << elapsed.count() << " ms." << std::endl;
    
    [self verifyResults];
}

- (void)verifyResults {
    uint8_t *resultBuffer = (uint8_t*)malloc(_width * _height * 4);
    MTLRegion region = MTLRegionMake2D(0, 0, _width, _height);
    [_outputTexture getBytes:resultBuffer bytesPerRow:_width * 4 fromRegion:region mipmapLevel:0];
    
    float totalBrightness = 0;
    float minBrightness = 255.0f;
    float maxBrightness = 0.0f;
    
    for (int i = 0; i < _width * _height * 4; i += 4) {
        float pixelBrightness = (resultBuffer[i] + resultBuffer[i+1] + resultBuffer[i+2]) / 3.0f;
        totalBrightness += pixelBrightness;
        if (pixelBrightness < minBrightness) {
            minBrightness = pixelBrightness;
        }
        if (pixelBrightness > maxBrightness) {
            maxBrightness = pixelBrightness;
        }
    }
    
    float averageBrightness = totalBrightness / (_width * _height);
    
    std::cout << "Average brightness of the output image: " << averageBrightness << std::endl;
    std::cout << "Minimum brightness of the output image: " << minBrightness << std::endl;
    std::cout << "Maximum brightness of the output image: " << maxBrightness << std::endl;
    
    free(resultBuffer);
}

- (void)getInputTextureData:(std::vector<uint8_t>&)imageData {
    MTLRegion region = MTLRegionMake2D(0, 0, _width, _height);
    [_inputTexture getBytes:imageData.data() bytesPerRow:_width * 4 fromRegion:region mipmapLevel:0];
}

@end

void compareResults(const std::vector<uint8_t>& gpuData, const std::vector<uint8_t>& cpuData, size_t width, size_t height) {
    if (gpuData.size() != cpuData.size()) {
        std::cerr << "Data size mismatch between GPU and CPU results." << std::endl;
        return;
    }

    size_t totalPixels = width * height;
    double totalDifference = 0.0;
    double maxDifference = 0.0;

    for (size_t i = 0; i < totalPixels * 4; i += 4) {
        double pixelDifference = 0.0;
        for (int j = 0; j < 4; ++j) { // Compare RGBA channels
            double diff = std::abs(static_cast<double>(gpuData[i + j]) - static_cast<double>(cpuData[i + j]));
            pixelDifference += diff;
        }
        totalDifference += pixelDifference;
        if (pixelDifference > maxDifference) {
            maxDifference = pixelDifference;
        }
    }

    double averageDifference = totalDifference / totalPixels;

    std::cout << "Average pixel difference: " << averageDifference << std::endl;
    std::cout << "Maximum pixel difference: " << maxDifference << std::endl;
}

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        
        if (!device) {
            std::cerr << "Metal is not supported on this device" << std::endl;
            return -1;
        }
        
        MetalCompute* metalCompute = [[MetalCompute alloc] initWithDevice:device];
        
        float brightness = 1.5f;  // Increase brightness by 50%
        
        // Process image on GPU
        [metalCompute processImageWithBrightness:brightness];
        
        // Prepare data for CPU processing
        std::vector<uint8_t> gpuImageData(metalCompute.width * metalCompute.height * 4);
        [metalCompute getInputTextureData:gpuImageData];
        
        // Process image on CPU
        std::vector<uint8_t> cpuImageData = gpuImageData; // Copy input data for CPU processing
        processImageOnCPU(cpuImageData, metalCompute.width, metalCompute.height, brightness);
        
        // Compare results
        compareResults(gpuImageData, cpuImageData, metalCompute.width, metalCompute.height);
    }
    
    return 0;
}
