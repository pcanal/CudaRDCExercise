# Build a static library with the "helper code".
add_library( HelperLib STATIC HelperLib/Macros.cuh
   HelperLib/HelperClass.cuh HelperLib/HelperClass.cu
   HelperLib/HelperClassPrinter.h HelperLib/HelperClassPrinter.cu )
set_target_properties( HelperLib PROPERTIES
   CUDA_SEPARABLE_COMPILATION ON
   POSITION_INDEPENDENT_CODE  ON )
set_property( TARGET HelperLib PROPERTY PUBLIC_HEADER
   HelperLib/Macros.cuh HelperLib/HelperClass.cuh
   HelperLib/HelperClassPrinter.h )
target_include_directories( HelperLib PUBLIC
   $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/HelperLib>
   $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> )

# Build a shared library with the "user code".
add_library( MultiplyLib SHARED MultiplyLib/ArrayMultiply.h
   MultiplyLib/ArrayMultiply.cu MultiplyLib/PrepareObject.h
   MultiplyLib/PrepareObject.cu MultiplyLib/Executor.cuh )
set_target_properties( MultiplyLib PROPERTIES
   CUDA_SEPARABLE_COMPILATION ON )
set_property( TARGET MultiplyLib PROPERTY PUBLIC_HEADER
   MultiplyLib/ArrayMultiply.h MultiplyLib/PrepareObject.h )
target_link_libraries( MultiplyLib PRIVATE HelperLib )
target_include_directories( MultiplyLib PUBLIC
   $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/MultiplyLib>
   $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> )

# Build an object library with the "user code".
add_library( MultiplyObjLib OBJECT MultiplyLib/ArrayMultiply.h
   MultiplyLib/ArrayMultiply.cu MultiplyLib/PrepareObject.h
   MultiplyLib/PrepareObject.cu MultiplyLib/Executor.cuh )
set_target_properties( MultiplyObjLib PROPERTIES
   CUDA_SEPARABLE_COMPILATION ON )
set_property( TARGET MultiplyObjLib PROPERTY PUBLIC_HEADER
   MultiplyLib/ArrayMultiply.h MultiplyLib/PrepareObject.h )
target_link_libraries( MultiplyObjLib PRIVATE HelperLib )
target_include_directories( MultiplyObjLib PUBLIC
   $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/MultiplyLib>
   $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> )

# Build a static library with the "user code".
add_library( MultiplyStatLib STATIC MultiplyLib/ArrayMultiply.h
   MultiplyLib/ArrayMultiply.cu MultiplyLib/PrepareObject.h
   MultiplyLib/PrepareObject.cu MultiplyLib/Executor.cuh )
set_target_properties( MultiplyStatLib PROPERTIES
   CUDA_SEPARABLE_COMPILATION ON
   POSITION_INDEPENDENT_CODE  ON )
set_property( TARGET MultiplyStatLib PROPERTY PUBLIC_HEADER
   MultiplyLib/ArrayMultiply.h MultiplyLib/PrepareObject.h )
target_link_libraries( MultiplyStatLib PRIVATE HelperLib )
target_include_directories( MultiplyStatLib PUBLIC
   $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/MultiplyLib>
   $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> )

# Build the test executable using the static and shared libraries.
add_executable( testComplexLink main.cxx )
target_link_libraries( testComplexLink PRIVATE
   MultiplyLib HelperLib )
set_property( TARGET testComplexLink PROPERTY INSTALL_RPATH
   "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}" )

# Build the test executable using the static and object libraries.
add_executable( testObjectLink main.cxx )
target_link_libraries( testObjectLink PRIVATE
   MultiplyObjLib HelperLib )

# Build the test executable using the two static libraries.
add_executable( testStaticLink main.cxx )
target_link_libraries( testStaticLink PRIVATE
   MultiplyStatLib HelperLib )

# Build the test executable using just the static library.
add_executable( testSimpleLink main.cxx
   MultiplyLib/ArrayMultiply.h MultiplyLib/ArrayMultiply.cu
   MultiplyLib/PrepareObject.h MultiplyLib/PrepareObject.cu
   MultiplyLib/Executor.cuh )
set_target_properties( testSimpleLink PROPERTIES
   CUDA_SEPARABLE_COMPILATION ON )
target_include_directories( testSimpleLink PRIVATE
   ${PROJECT_SOURCE_DIR}/MultiplyLib )
target_link_libraries( testSimpleLink PRIVATE HelperLib )

# Ctest the executables
if(BUILD_TESTING)
  add_test(NAME atl_testComplexLink COMMAND testComplexLink)
  add_test(NAME atl_testObjectLink COMMAND testObjectLink)
  add_test(NAME atl_testStaticLink COMMAND testStaticLink)
  add_test(NAME atl_testSimpleLink COMMAND testSimpleLink)
endif()