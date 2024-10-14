cmake_minimum_required(VERSION 3.10)
project(CudaPrepCPP VERSION 1.0 LANGUAGES CXX)

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

# Source files
set(SOURCES
    main.cpp
    ${CMAKE_SOURCE_DIR}/src/newsimreader.cpp
    ${CMAKE_SOURCE_DIR}/src/funcs.cpp
    ${CMAKE_SOURCE_DIR}/src/overloads.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu_kernels.cpp
)

# Header files
set(HEADERS
    ${CMAKE_SOURCE_DIR}/src/newsimreader.h
    ${CMAKE_SOURCE_DIR}/src/funcs.h
    ${CMAKE_SOURCE_DIR}/src/overloads.h
    ${CMAKE_SOURCE_DIR}/src/cuda_replacements.h
    ${CMAKE_SOURCE_DIR}/src/Variable.h
    ${CMAKE_SOURCE_DIR}/src/cpu_kernels.h
    ${CMAKE_SOURCE_DIR}/src/datatemplate.h
)

# Create executable
add_executable(CudaPrepCPP ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(CudaPrepCPP PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/demo_creator/include
)

# Link libraries
target_link_libraries(CudaPrepCPP PRIVATE 
    nlohmann_json::nlohmann_json
    demo_creator
)

# Set compiler flags
if(MSVC)
    target_compile_options(CudaPrepCPP PRIVATE /W4)
else()
    target_compile_options(CudaPrepCPP PRIVATE -Wall -Wextra -pedantic)
endif()

# Enable optimization for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Set output directory
set_target_properties(CudaPrepCPP PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Install target
install(TARGETS CudaPrepCPP DESTINATION bin)
