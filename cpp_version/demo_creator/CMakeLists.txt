cmake_minimum_required(VERSION 3.10)
project(demo_creator)

set(CMAKE_CXX_STANDARD 17)

# Add the path to the Variable.h file
include_directories(${CMAKE_SOURCE_DIR}/src)

# Add the demo_creator library
add_library(demo_creator
    src/demo_creator.cpp
    include/demo_creator.h
)

# Link against nlohmann_json
target_link_libraries(demo_creator PRIVATE nlohmann_json::nlohmann_json)

# Add include directories
target_include_directories(demo_creator PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)