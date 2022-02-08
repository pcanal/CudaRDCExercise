#include "foo.h"

#include <cstdio>
#include <iostream>
#include <stdexcept>

__global__ void fooKernel()
{
    printf("Running fooKernel\n");
}

void foo()
{
    fooKernel<<<1,1>>>();
    cudaError_t cuda_result_ = cudaPeekAtLastError();
    if (cuda_result_ != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(cuda_result_));
    }
    cudaDeviceSynchronize();
}