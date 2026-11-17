cmake_minimum_required(VERSION 3.25)
project(CPPNeuralNetwork VERSION 1.1.0 LANGUAGES CXX)

# -------------------------------------------------------------------------------------------------
#   LANGUAGE A& TOOLCHAIN
# -------------------------------------------------------------------------------------------------
# C++26: the project tracks the newest standard on purpose (the code itself only needs
# C++20 features so far, but the toolchain bar stays current).
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Eigen is compiled into every TU that includes our headers, and it is only fast with
# optimizations on. A build without a type would silently produce a 10-100x slower
# binary, so default to Release.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                 Debug Release RelWithDebInfo MinSizeRel)
endif()

# Useful for clangd / VSCode IntelliSense.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------------------------------------------------------------------------------------
#   WARNINGS & SANITIZERS
# -------------------------------------------------------------------------------------------------
function(nn_apply_warnings target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
    endif()
endfunction()

option(ENABLE_SANITIZERS "Enable AddressSanitizer + UBSan (debug builds)" OFF)
function(nn_apply_sanitizers target)
    if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
endfunction()

# -------------------------------------------------------------------------------------------------
#   FRAMEWORK: REUSABLE STATIC LIBRARY
# -------------------------------------------------------------------------------------------------
add_library(neural STATIC
    src/Activation.cpp
    src/DenseLayer.cpp
    src/Loss.cpp
    src/Network.cpp
    src/Optimizer.cpp
)

# Eigen is header-only and appears in our public headers, so it must be propagated to
# consumers as well (bundled as a git submodule: no find_package needed, the build
# stays self-contained).
target_include_directories(neural
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen
)

nn_apply_warnings(neural)
nn_apply_sanitizers(neural)

# -------------------------------------------------------------------------------------------------
#   EXAMPLES & TESTS
# -------------------------------------------------------------------------------------------------
add_subdirectory(examples)
add_subdirectory(tests)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME activations_loss COMMAND nn_test_activations)
    add_test(NAME gradient_check COMMAND nn_test_gradients)
    add_test(NAME training COMMAND nn_test_training)
endif()
