# RDC form of the ATLAS link example
# 1. HelperLib "base" library
# 2. MultiplyLib "user" library, depends on HelperLib
# 3. Executable that consumes both libraries

cuda_rdc_add_library(RDCHelperLib SHARED 
  HelperLib/Macros.cuh
  HelperLib/HelperClass.cuh HelperLib/HelperClass.cu
  HelperLib/HelperClassPrinter.h HelperLib/HelperClassPrinter.cu )
cuda_rdc_target_include_directories(RDCHelperLib PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/HelperLib>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> )

cuda_rdc_add_library(RDCMultiplyLib SHARED
   MultiplyLib/ArrayMultiply.h
   MultiplyLib/ArrayMultiply.cu MultiplyLib/PrepareObject.h
   MultiplyLib/PrepareObject.cu MultiplyLib/Executor.cuh )
cuda_rdc_target_include_directories( RDCMultiplyLib PUBLIC
   $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/MultiplyLib>
   $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> )
cuda_rdc_target_link_libraries(RDCMultiplyLib PRIVATE RDCHelperLib)

# CPP only consumer
add_executable(testRDCLinkCPP main.cxx)
cuda_rdc_target_link_libraries(testRDCLinkCPP PRIVATE RDCMultiplyLib RDCHelperLib)

# CPP+CUDA consumer
add_executable(testRDCLinkCUDA main.cxx dummy.cu)
set_target_properties(testRDCLinkCUDA PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
cuda_rdc_target_link_libraries(testRDCLinkCUDA PRIVATE RDCMultiplyLib RDCHelperLib)

if(BUILD_TESTING)
  add_test(NAME atl_testRDCLinkCPP COMMAND testRDCLinkCPP)
  add_test(NAME atl_testRDCLinkCUDA COMMAND testRDCLinkCUDA)
endif()
