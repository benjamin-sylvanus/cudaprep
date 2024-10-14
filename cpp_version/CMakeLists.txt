cmake_minimum_required(VERSION 3.10)
project(CudaPrepCPP VERSION 1.0 LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the nlohmann/json library
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

# Add the demo_creator subdirectory
add_subdirectory(demo_creator)

# Source files for the main executable
set(SOURCES
    main.cpp
    ${CMAKE_SOURCE_DIR}/src/newsimreader.cpp
    ${CMAKE_SOURCE_DIR}/src/funcs.cpp
    ${CMAKE_SOURCE_DIR}/src/overloads.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu_kernels.cpp
)

# Header files for the main executable
set(HEADERS
    ${CMAKE_SOURCE_DIR}/src/newsimreader.h
    ${CMAKE_SOURCE_DIR}/src/funcs.h
    ${CMAKE_SOURCE_DIR}/src/overloads.h
    ${CMAKE_SOURCE_DIR}/src/cuda_replacements.h
    ${CMAKE_SOURCE_DIR}/src/Variable.h
    ${CMAKE_SOURCE_DIR}/src/cpu_kernels.h
    ${CMAKE_SOURCE_DIR}/src/datatemplate.h
)

# Create main executable
add_executable(CudaPrepCPP ${SOURCES} ${HEADERS})

# Include directories for the main executable
target_include_directories(CudaPrepCPP PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/demo_creator/include
)

# Link libraries for the main executable
target_link_libraries(CudaPrepCPP PRIVATE 
    nlohmann_json::nlohmann_json
    demo_creator
)

# Set compiler flags for the main executable
if(MSVC)
    target_compile_options(CudaPrepCPP PRIVATE /W4 /O2)
else()
    target_compile_options(CudaPrepCPP PRIVATE -Wall -Wextra -pedantic -O3)
endif()

# Set output directory for the main executable
set_target_properties(CudaPrepCPP PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Install target for the main executable
install(TARGETS CudaPrepCPP DESTINATION bin)

# Metal-based Vector Addition executable
if(APPLE)
    # Source files for the Metal-based executable
    set(METAL_SOURCES
        ${CMAKE_SOURCE_DIR}/metal-src/metal-main.mm
        ${CMAKE_SOURCE_DIR}/metal-src/cpu_kernels.cpp
    )

    # Create Metal-based executable
    add_executable(MetalVectorAdd ${METAL_SOURCES})

    # Set Objective-C++ as the language for the metal-main.mm file
    set_source_files_properties(${CMAKE_SOURCE_DIR}/metal-src/metal-main.mm PROPERTIES LANGUAGE OBJCXX)

    # Include directories for the Metal-based executable
    target_include_directories(MetalVectorAdd PRIVATE 
        ${CMAKE_SOURCE_DIR}/metal-src
    )

    # Link Metal framework
    target_link_libraries(MetalVectorAdd PRIVATE
        "-framework Metal"
        "-framework Foundation"
        "-framework MetalKit"
    )

    # Set compiler flags for the Metal-based executable
    target_compile_options(MetalVectorAdd PRIVATE -O3)

    # Set output directory for the Metal-based executable
    set_target_properties(MetalVectorAdd PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # Copy the Metal shader source to the build directory
    configure_file(${CMAKE_SOURCE_DIR}/metal-src/Shaders.metal ${CMAKE_BINARY_DIR}/bin/Shaders.metal COPYONLY)

    # Install target for the Metal-based executable
    install(TARGETS MetalVectorAdd DESTINATION bin)
    install(FILES ${CMAKE_SOURCE_DIR}/metal-src/Shaders.metal DESTINATION bin)
else()
    message(STATUS "Not building on Apple platform. Skipping Metal-based executable.")
endif()
