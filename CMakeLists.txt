cmake_minimum_required(VERSION 3.10)
project(CPPNeuralNetwork)

# Check and force C++ 17 or higher.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# This is useful for autoincluding directories.
include_directories(${CMAKE_SOURCE_DIR}/external/eigen)
include_directories(${CMAKE_SOURCE_DIR}/include)

# Creates main exe.
add_executable(main
    src/Loss.cpp
    src/Layer.cpp
    src/Network.cpp
    src/Main.cpp
)