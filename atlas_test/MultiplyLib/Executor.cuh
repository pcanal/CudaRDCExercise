// Dear emacs, this is -*- c++ -*-
#ifndef TEST11_EXECUTOR_CUH
#define TEST11_EXECUTOR_CUH

// Local include(s).
#include "HelperClass.cuh"
#include "Macros.cuh"

// CUDA include(s).
#include <cuda.h>

namespace {

   /// A function mimicking somethig a bit more complicated from my
   /// original code.
   std::size_t toArrayId( const char* number ) {
      return std::atoi( number );
   }

} // private namespace

/// Function executing the functor on the host
template< class FUNCTOR, typename... IDS >
__host__
void hostExecute( HelperClass& helper, IDS... ids ) {

   /// Instantiate the functor.
   auto functor = FUNCTOR();

   // Call it on every element of the helper object.
   for( std::size_t i = 0; i < helper.size(); ++i ) {
      functor( i, helper, ids... );
   }
   return;
}

/// Kernel executing the functor on the device
template< class FUNCTOR, typename... IDS >
__global__
void deviceExecute( std::size_t csize, std::size_t vsize, float** arrays,
                    IDS... ids ) {

   // Find the current index that we need to process.
   const int index = blockIdx.x * blockDim.x + threadIdx.x;
   if( index >= csize ) {
      return;
   }

   // Construct the helper object.
   HelperClass helper( csize, vsize, arrays );

   // Execute the specified functor.
   FUNCTOR()( index, helper, ids... );
   return;
}

/// Function to be called in user code to execute a user defined functor on
/// a helper object's payload.
template< class FUNCTOR, typename... NAMES >
__host__
void execute( HelperClass& helper, NAMES... names ) {

   // Check if any device is available at runtime. Note that I'm not checking
   // for a return code, since the call will fail when no devices are available.
   // And that's okay.
   int nDevices = 0;
   cudaGetDeviceCount( &nDevices );

   // If no devices are available, run the code on the host.
   if( nDevices == 0 ) {
      hostExecute< FUNCTOR >( helper, toArrayId( names )... );
      return;
   }

   // If we are here, a device *is* available...

   // Create a stream for the execution.
   cudaStream_t stream;
   CUDA_CHECK( cudaStreamCreate( &stream ) );

   // Execute the kernel.
   const int nThreadsPerBlock = 256;
   const int n = static_cast< int >( helper.size() );
   const int nBlocks = ( n + nThreadsPerBlock - 1 ) / nThreadsPerBlock;
   auto arrays = helper.arrays();
   deviceExecute< FUNCTOR ><<< nBlocks, nThreadsPerBlock, 0, stream >>>(
      n, arrays.first, arrays.second, toArrayId( names )... );

   // Wait for the calculation to finish.
   CUDA_CHECK( cudaGetLastError() );
   CUDA_CHECK( cudaStreamSynchronize( stream ) );
   CUDA_CHECK( cudaStreamDestroy( stream ) );
   return;
}

#endif // TEST11_EXECUTOR_CUH
