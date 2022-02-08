cmake_minimum_required(VERSION 3.18...3.22)
project(cuda-rdc-test C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
include(CTest)
include(CudaRdcUtils.cmake)

# Simplest test case
add_subdirectory(simple_test)

# ATLAS/CMS test case
add_subdirectory(atlas_test)