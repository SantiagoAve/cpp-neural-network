cmake_minimum_required(VERSION 3.10)
project(CPPNeuralNetwork)

# Check and force C++ 26 (Or higher in the future).
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# This is useful for autoincluding directories.
include_directories(${CMAKE_SOURCE_DIR}/external/eigen)
include_directories(${CMAKE_SOURCE_DIR}/include)

# Creates main exe.
add_executable(main
    src/Loss.cpp
    src/Activation.cpp
    src/Layer.cpp
    src/Network.cpp
    src/Optimizer.cpp
    src/Main.cpp
)