// Dear emacs, this is -*- c++ -*-

// Local include(s).
#include "HelperClass.cuh"
#include "Executor.cuh"

namespace {

   /// Functor multiplying a specific array's elements by 1.5
   class ArrayMultiplier {
   public:
      __device__ __host__
      void operator()( std::size_t i, HelperClass& helper, std::size_t id ) {
         helper.array( id )[ i ] *= 1.5f;
         return;
      }
   }; // class ArrayMultiplier

} // private namespace

#if defined(_WIN32) || defined(_WIN64)
__declspec(dllexport)
#endif
void
#if defined(_WIN32) || defined(_WIN64)
__cdecl
#endif
arrayMultiply( HelperClass& helper ) {

   execute< ::ArrayMultiplier >( helper, "0" );
   return;
}
