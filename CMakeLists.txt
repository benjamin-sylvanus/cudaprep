cmake_minimum_required(VERSION 3.24)
project(cudaprep)

set(CMAKE_CXX_STANDARD 23)

add_executable(cudaprep main.cpp src/simulation.cpp src/simulation.h src/simreader.cpp src/simreader.h src/datatemplate.cpp src/datatemplate.h)
