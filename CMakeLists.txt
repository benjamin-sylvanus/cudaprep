cmake_minimum_required(VERSION 3.18)
project(your_project LANGUAGES CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the nlohmann/json library
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

find_package(CUDA REQUIRED)
set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
find_library(CUBLAS_LIBRARY cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(CURAND_LIBRARY curand HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(CUSPARSE_LIBRARY cusparse HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

include_directories(${CMAKE_SOURCE_DIR})


set(SOURCES
   ${CMAKE_SOURCE_DIR}/src/datatemplate.cpp
   ${CMAKE_SOURCE_DIR}/src/simulation.cpp
   ${CMAKE_SOURCE_DIR}/src/simreader.cpp
   ${CMAKE_SOURCE_DIR}/src/controller.cpp
   ${CMAKE_SOURCE_DIR}/src/viewer.cpp
   ${CMAKE_SOURCE_DIR}/src/funcs.cpp
   ${CMAKE_SOURCE_DIR}/src/overloads.cpp
)

set(OBJECTS "")
foreach(SRC_FILE ${SOURCES})
    get_filename_component(FILE_NAME ${SRC_FILE} NAME_WE)
    set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/bin/${FILE_NAME}.o")
    add_custom_command(
        OUTPUT ${OBJ_FILE}
        COMMAND ${CUDA_NVCC_EXECUTABLE}
        ARGS -x cu -arch=sm_75 -I${CMAKE_SOURCE_DIR} -dc ${SRC_FILE} -o ${OBJ_FILE}
        DEPENDS ${SRC_FILE}
    )
    list(APPEND OBJECTS ${OBJ_FILE})
endforeach()

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/bin/main.o
    COMMAND ${CUDA_NVCC_EXECUTABLE}
    ARGS -arch=sm_75 -ccbin ${CMAKE_CXX_COMPILER} -dc -L${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CMAKE_SOURCE_DIR}/temp.cu -o ${CMAKE_CURRENT_BINARY_DIR}/bin/main.o
    DEPENDS ${CMAKE_SOURCE_DIR}/temp.cu
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/bin/gpuCode.o
    COMMAND ${CUDA_NVCC_EXECUTABLE}
    ARGS -arch=sm_75 -dlink ${OBJECTS} ${CMAKE_CURRENT_BINARY_DIR}/bin/main.o -o ${CMAKE_CURRENT_BINARY_DIR}/bin/gpuCode.o
    DEPENDS ${OBJECTS} ${CMAKE_CURRENT_BINARY_DIR}/bin/main.o
)

add_executable(app ${CMAKE_CURRENT_BINARY_DIR}/bin/main.o ${CMAKE_CURRENT_BINARY_DIR}/bin/gpuCode.o ${OBJECTS})
set_target_properties(app PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(app ${CUBLAS_LIBRARY} ${CURAND_LIBRARY} ${CUSPARSE_LIBRARY} ${CUDA_LIBRARIES} nlohmann_json::nlohmann_json)