
# Set up the project.
cmake_minimum_required(VERSION 3.18...3.22)
project( CUDALinkTest VERSION 1.0.0 LANGUAGES CXX CUDA )

# Use C++14.
set( CMAKE_CXX_STANDARD 14 CACHE STRING
   "The C++ version to use" )
set( CMAKE_CXX_EXTENSIONS OFF CACHE BOOL
   "(Dis)Allow C++ compiler extensions" )

# Use the GNU default directory names for installation.
include( GNUInstallDirs )

# Native build
include(build_native.cmake)

# RDC build
include(build_rdc.cmake)
