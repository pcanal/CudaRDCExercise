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

All tests except `Test #2: test_shar_foo_CUDA_Consumer` should pass. The failing
test simply tries to call the `foo()` function, which in turn calls the `fooKernel`
kernel. The failure message is "invalid device function", but all device linking appears
to have been done correctly. The only difference between Test 2 and Test 1 is that 
the former includes its own CUDA "code" (an empty file to trigger device compilation/linking)
but this does not call any device code in foo and so should be considered a separate,
independent device link.