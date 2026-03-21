cmake_minimum_required(VERSION 3.10)
project(CPPNeuralNetwork)

# Esto es para que use C++ versión 17 o más.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Esto es pra incluir a Eigen desde el submódulo.
include_directories(${CMAKE_SOURCE_DIR}/external/eigen)

# Y esto va a crear el ejecutable principal.
add_executable(main src/main.cpp)