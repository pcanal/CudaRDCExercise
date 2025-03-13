cmake_minimum_required(VERSION 3.18...3.22)
project(cuda-rdc-test C CXX CUDA)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 14)
include(CTest)
include(CudaRdcUtils.cmake)

option(CUDA_RDC_UTILS_TESTING "Enable tesing of CudaRdcUtils.cmake" OFF)

if(CUDA_RDC_UTILS_TESTING)
  # Simplest test case
  add_subdirectory(simple_test)

  # Simplest module test case
  add_subdirectory(module_test)

  # ATLAS/CMS test case
  add_subdirectory(atlas_test)
endif()
