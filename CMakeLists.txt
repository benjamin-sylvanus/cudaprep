cmake_minimum_required(VERSION 3.24)
project(cudaprep)

set(CMAKE_CXX_STANDARD 23)

add_executable(cudaprep main.cu src/simulation.cpp src/simulation.h src/simreader.cpp src/simreader.h src/datatemplate.h src/funcs.cpp src/funcs.h src/viewer.cpp src/viewer.h src/controller.h src/controller.cpp)
