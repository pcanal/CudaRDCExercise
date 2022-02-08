#! /bin/bash

# path to the source code
SRC=$PWD

CXXFLAGS="-std=c++14 -O2 -pthread -fPIC"

CUDA_BASE=${CUDA_HOME}
CUDA_CXXFLAGS="-std=c++14 -O2 -gencode arch=compute_50,code=sm_50 --cudart=shared"
CUDA_LDFLAGS="-L$CUDA_BASE/lib64/stubs -L$CUDA_BASE/lib64 -lcudart -lcudadevrt -lcuda"

rm -rf tmp lib
mkdir tmp lib
set -ex

# build libHelperLib.so
$CUDA_BASE/bin/nvcc -dc -I$SRC/HelperLib -I$CUDA_BASE/include $CUDA_CXXFLAGS --compiler-options "$CXXFLAGS" $SRC/HelperLib/HelperClassPrinter.cu -o tmp/HelperClassPrinter.cu.o
$CUDA_BASE/bin/nvcc -dc -I$SRC/HelperLib -I$CUDA_BASE/include $CUDA_CXXFLAGS --compiler-options "$CXXFLAGS" $SRC/HelperLib/HelperClass.cu -o tmp/HelperClass.cu.o
$CUDA_BASE/bin/nvcc -dlink -Ltmp -Llib $CUDA_CXXFLAGS $CUDA_LDFLAGS --compiler-options "$CXXFLAGS" tmp/HelperClassPrinter.cu.o tmp/HelperClass.cu.o -o tmp/HelperLib_cudadlink.o
g++ -shared $CXXFLAGS tmp/HelperClassPrinter.cu.o tmp/HelperClass.cu.o tmp/HelperLib_cudadlink.o -Llib -Ltmp $CUDA_LDFLAGS -o lib/libHelperLib.so

# build libHelperLib_nv.a
objcopy -j ".nv*" -j "__nv*" tmp/HelperClassPrinter.cu.o tmp/HelperClassPrinter.cu_nv.o
objcopy -j ".nv*" -j "__nv*" tmp/HelperClass.cu.o tmp/HelperClass.cu_nv.o
ar crs tmp/libHelperLib_nv.a tmp/HelperClassPrinter.cu_nv.o tmp/HelperClass.cu_nv.o

# build libMultiplyLib.so
$CUDA_BASE/bin/nvcc -dc -I$SRC/HelperLib -I$SRC/MultiplyLib -I$CUDA_BASE/include $CUDA_CXXFLAGS --compiler-options "$CXXFLAGS" $SRC/MultiplyLib/ArrayMultiply.cu -o tmp/ArrayMultiply.cu.o
$CUDA_BASE/bin/nvcc -dc -I$SRC/HelperLib -I$SRC/MultiplyLib -I$CUDA_BASE/include $CUDA_CXXFLAGS --compiler-options "$CXXFLAGS" $SRC/MultiplyLib/PrepareObject.cu -o tmp/PrepareObject.cu.o
$CUDA_BASE/bin/nvcc -dlink -Ltmp -lHelperLib_nv -Llib $CUDA_CXXFLAGS $CUDA_LDFLAGS --compiler-options "$CXXFLAGS" tmp/ArrayMultiply.cu.o tmp/PrepareObject.cu.o -o tmp/MultiplyLib_cudadlink.o
g++ -shared $CXXFLAGS tmp/ArrayMultiply.cu.o tmp/PrepareObject.cu.o tmp/MultiplyLib_cudadlink.o -Llib -Ltmp -lHelperLib_nv -lHelperLib $CUDA_LDFLAGS -o lib/libMultiplyLib.so

# build libMultiplyLib_nv.a
objcopy -j ".nv*" -j "__nv*" tmp/ArrayMultiply.cu.o tmp/ArrayMultiply.cu_nv.o
objcopy -j ".nv*" -j "__nv*" tmp/PrepareObject.cu.o tmp/PrepareObject.cu_nv.o
ar crs tmp/libMultiplyLib_nv.a tmp/ArrayMultiply.cu_nv.o tmp/PrepareObject.cu_nv.o

# build the main executable
g++ -c -I$SRC/HelperLib -I$SRC/MultiplyLib -I$CUDA_BASE/include $CXXFLAGS $SRC/main.cxx -o tmp/main.cxx.o
g++ $CXXFLAGS tmp/main.cxx.o -Llib -lMultiplyLib -lHelperLib $CUDA_LDFLAGS -Wl,-rpath,$PWD/lib -o testComplexLink