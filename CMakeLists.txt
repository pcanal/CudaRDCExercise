cmake_minimum_required(VERSION 3.18...3.22)
project(cuda-rdc-test C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
include(CTest)
include(CudaRdcUtils.cmake)

# 1a. Build "shar_foo" as a full, shared RDC library
cuda_rdc_add_library(shar_foo SHARED foo.h foo.cu)

# 1b. Build "stat_foo" as a full, static RDC Library 
cuda_rdc_add_library(stat_foo STATIC foo.h foo.cu)

# 1c. Consume "shar_foo" in a pure C++ exe
add_executable(shar_foo_CPP_Consumer fooConsumer.cc)
cuda_rdc_target_link_libraries(shar_foo_CPP_Consumer PRIVATE shar_foo)
add_test(NAME test_shar_foo_CPP_Consumer COMMAND shar_foo_CPP_Consumer)

# 1d. Consume "shar_foo" in a mixed C++/Cuda exe
add_executable(shar_foo_CUDA_Consumer fooConsumer.cc dummy.cu)
set_target_properties(shar_foo_CUDA_Consumer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
cuda_rdc_target_link_libraries(shar_foo_CUDA_Consumer PRIVATE shar_foo)
# As implemented, the consuming executable *must* use the Shared CUDA runtime because shar_foo does
set_target_properties(shar_foo_CUDA_Consumer PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
add_test(NAME test_shar_foo_CUDA_Consumer COMMAND shar_foo_CUDA_Consumer)

# 1e. Consume "stat_foo" in a pure C++ exe
add_executable(stat_foo_CPP_Consumer fooConsumer.cc)
cuda_rdc_target_link_libraries(stat_foo_CPP_Consumer PRIVATE stat_foo)
add_test(NAME test_stat_foo_CPP_Consumer COMMAND stat_foo_CPP_Consumer)

# 1f. Consume "stat_foo" in a mixed C++/Cuda exe
add_executable(stat_foo_CUDA_Consumer fooConsumer.cc dummy.cu)
cuda_rdc_target_link_libraries(stat_foo_CUDA_Consumer PRIVATE stat_foo)
add_test(NAME test_stat_foo_CUDA_Consumer COMMAND stat_foo_CUDA_Consumer)

# 2. What happens if we build shared libs in mixed case
#     This is to try and investigate why 1d. above does not work in the RDC case
# a. Native CMake Cuda lib
add_library(fooCMakeNativeCuda SHARED foo.h foo.cu)
set_target_properties(fooCMakeNativeCuda 
  PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    # Though we don't require it, what happens if we mix Shared here, and static in the exe in 1d.
    CUDA_RUNTIME_LIBRARY Shared
  )
add_executable(fooCMakeNativeCuda_CPP_Consumer fooConsumer.cc)
target_link_libraries(fooCMakeNativeCuda_CPP_Consumer PRIVATE fooCMakeNativeCuda)
add_test(NAME test_native_CPP_Consumer COMMAND fooCMakeNativeCuda_CPP_Consumer)

add_executable(fooCMakeNativeCuda_CUDA_Consumer fooConsumer.cc dummy.cu)
# Force device linking
set_target_properties(fooCMakeNativeCuda_CUDA_Consumer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(fooCMakeNativeCuda_CUDA_Consumer PRIVATE fooCMakeNativeCuda)
add_test(NAME test_native_CUDA_Consumer COMMAND fooCMakeNativeCuda_CUDA_Consumer)
