# CUDA RDC Test

Noddy exercise of the CudaRDCUtil module to confirm/document behaviour and
requirements for building/consuming libraries built this way.

It just needs CMake 3.18 or newer, C++ and CUDA compilers (tested with CMake
3.18, GCC 10.2 and CUDA 11.1.1). To build/run:

```
$ cmake -S. -Bbuild
$ cmake --build build
$ cmake --build build --target test
```

## `simple_test`

A very simple demo of `CudaRdcUtils` and comparison with reproducing it in native CMake

All tests except `test_shar_foo_CUDA_Consumer` should pass. The failing
test simply tries to call the `foo()` function, which in turn calls the `fooKernel`
kernel. The failure message is "invalid device function", but all device linking appears
to have been done correctly. The only difference between Test 2 and Test 1 is that 
the former includes its own CUDA "code" (an empty file to trigger device compilation/linking)
but this does not call any device code in foo and so should be considered a separate,
independent device link.

*Solution*: the `shar_foo_CUDA_Consumer` executable _must_ link to the shared CUDA runtime
library. By default, CMake links a CUDA-using executable using the static runtime. In the
`shar_foo_CUDA_Consumer` case, it links to `libshar_foo.so` which links to the shared runtime,
but the device link file and the final exe use the static runtime. This seems to be the cause of
the error. Setting `shar_foo_CUDA_Consumer`'s `CUDA_RUNTIME_LIBRARY` target property to `Shared`
results in a running executable.

# atlas_test

A test case of CudaRdcUtils with an example linking problem from ATLAS (with input from CMS)
The original project and code are at: https://github.com/krasznaa/CUDALinkTest.git.

It's adapted as follows:

- The original build is factored into the `build_native.cmake` script
- Install commands removed for now
- CudaRdcUtils form of build included in `build_rdc.cmake` script
  - Builds both `HelperLib` and `MultiplyLib` as RDC shared libraries
  - Links them to two forms of the consuming application
    1. A pure C++ consumer
    2. A mixed C++/CUDA consumer (CUDA is dummy, but sufficient to trigger device linking step)

Everything should compile fine, but we observe that `atl_testComplexLink` fails. This is
expected from the upstream project. The RDC adaption here however fails the `atl_testRDCLinkCUDA`
test with an error:

```console
$ ./atlas_test/testRDCLinkCUDA 
./atlas_test/testRDCLinkCUDA: symbol lookup error: <BUILD_DIR>/atlas_test/libRDCHelperLib.so: undefined symbol: __cudaRegisterLinkedBinary_46_tmpxft_00000797_00000000_7_HelperClass_cpp1_ii_c5c4d760
```

It's compiled/linked using the commands:

```console
[ 75%] Building CXX object atlas_test/CMakeFiles/testRDCLinkCUDA.dir/main.cxx.o
cd <BUILD_DIR>/atlas_test && c++  -I<SRC_DIR>/atlas_test/MultiplyLib -I<SRC_DIR>/atlas_test/HelperLib -std=c++14 -o CMakeFiles/testRDCLinkCUDA.dir/main.cxx.o -c <SRC_DIR>/atlas_test/main.cxx
[ 83%] Building CUDA object atlas_test/CMakeFiles/testRDCLinkCUDA.dir/dummy.cu.o
cd <BUILD_DIR>/atlas_test && nvcc -forward-unknown-to-host-compiler  -I<SRC_DIR>/atlas_test/MultiplyLib -I<SRC_DIR>/atlas_test/HelperLib --generate-code=arch=compute_61,code=[compute_61,sm_61] -std=c++14 -x cu -dc <SRC_DIR>/atlas_test/dummy.cu -o CMakeFiles/testRDCLinkCUDA.dir/dummy.cu.o
[ 91%] Linking CUDA device code CMakeFiles/testRDCLinkCUDA.dir/cmake_device_link.o
nvcc -forward-unknown-to-host-compiler  --generate-code=arch=compute_61,code=[compute_61,sm_61] -Xcompiler=-fPIC -Wno-deprecated-gpu-targets -shared -dlink CMakeFiles/testRDCLinkCUDA.dir/main.cxx.o CMakeFiles/testRDCLinkCUDA.dir/dummy.cu.o -o CMakeFiles/testRDCLinkCUDA.dir/cmake_device_link.o   -L<CUDA_HOME>/lib/stubs  -lcudadevrt -lcudart_static -lrt -lpthread -ldl  -L"<CUDA_HOME>/lib"
[100%] Linking CXX executable testRDCLinkCUDA
c++ CMakeFiles/testRDCLinkCUDA.dir/main.cxx.o CMakeFiles/testRDCLinkCUDA.dir/dummy.cu.o CMakeFiles/testRDCLinkCUDA.dir/cmake_device_link.o -o testRDCLinkCUDA   -L<CUDA_HOME>/lib/stubs  -Wl,-rpath,<BUILD_DIR>/atlas_test libRDCMultiplyLib.so libRDCHelperLib.so -lcudadevrt -lcudart_static -lrt -lpthread -ldl
```

It was observed before that using the static CUDA runtime could cause issues, but changing this in the CMake
script leads to a link error:

```console
[ 83%] Linking CXX executable testRDCLinkCUDA
c++ CMakeFiles/testRDCLinkCUDA.dir/main.cxx.o CMakeFiles/testRDCLinkCUDA.dir/dummy.cu.o CMakeFiles/testRDCLinkCUDA.dir/cmake_device_link.o -o testRDCLinkCUDA   -L<CUDA_HOME>/lib/stubs  -Wl,-rpath,<BUILD_DIR>/atlas_test libRDCMultiplyLib.so libRDCHelperLib.so -lcudadevrt -lcudart 
libRDCHelperLib.so: error: undefined reference to '__cudaRegisterLinkedBinary_46_tmpxft_0000204b_00000000_7_HelperClass_cpp1_ii_c5c4d760'
libRDCMultiplyLib.so: error: undefined reference to '__cudaRegisterLinkedBinary_48_tmpxft_0000238a_00000000_7_PrepareObject_cpp1_ii_6915bc6a'
libRDCHelperLib.so: error: undefined reference to '__cudaRegisterLinkedBinary_53_tmpxft_00002089_00000000_7_HelperClassPrinter_cpp1_ii_cd041c86'
libRDCMultiplyLib.so: error: undefined reference to '__cudaRegisterLinkedBinary_48_tmpxft_00002340_00000000_7_ArrayMultiply_cpp1_ii_ae303cef'
collect2: error: ld returned 1 exit status
```

The main observation here is that we aren't including the Helper/Multiply static libraries in the device link step
though they are nominally needed as we end up linking to the raw shared libraries which are not device linked. In
the pure CPP case, we link to the `_final` device-linked shared libraries and all is fine.

This suggests either a mis-use of the RDC functions, or a bug in how they resolve what to include in device links.

See also the [`build_cmssw.sh` script](atlas_test/build_cmssw.sh) which outlines CMSSW's method for handling CUDA
libraries (originally posted [here](https://github.com/krasznaa/CUDALinkTest/issues/1)). This is very similar to
the RDC pattern, the `_nv` libraries taking the place of the RDC `_static` ones in the process. Whilst the `_nv` libs
are stripped down to only device/CUDA symbols, this shouldn't make too much difference when device linking.
