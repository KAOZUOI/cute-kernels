cmake_minimum_required(VERSION 3.18)
project(cute-gemm LANGUAGES CXX CUDA)

include(CheckLanguage)
check_language(CXX)
if(CMAKE_CXX_COMPILER)
    enable_language(CXX)
else()
    message(FATAL_ERROR "No C++ support")
endif()

if(CMAKE_COMPILER_IS_GNUCC)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.3.0)
        message(FATAL_ERROR "gcc >= 13.3.0 is required.")
    endif()
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(CUDAToolkit_VERSION VERSION_LESS "12.4")
    message(FATAL_ERROR "CUDA Version >= 12.4 is required (found ${CUDAToolkit_VERSION}).")
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;90a")
endif()

add_compile_options(-ffunction-sections -fdata-sections -fno-common)
add_compile_options(-fPIC)
add_compile_options("$<$<CONFIG:DEBUG>:-O0;-g>")
add_compile_options("$<$<CONFIG:RELEASE>:-O3;-g>")

add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=--warn-on-spills>")

add_link_options("LINKER:--as-needed")
add_link_options("LINKER:--gc-sections")

find_package(Thrust QUIET)
if(Thrust_FOUND)
    thrust_create_target(thrust)
    set_property(GLOBAL PROPERTY GLOBAL_THRUST_TARGET thrust)
else()
    message(STATUS "Using Thrust from CUDA toolkit")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/gemm/include
)

add_definitions(-DCUTLASS_ARCH_MMA_SM90A_SUPPORTED=1)
add_definitions(-DCUTLASS_ARCH_SM80_SUPPORTED=1)
add_definitions(-DCUTLASS_ENABLE_ASYNC_COPY=1)
add_subdirectory(gemm)