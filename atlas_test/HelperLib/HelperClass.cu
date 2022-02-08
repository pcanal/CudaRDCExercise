// Dear emacs, this is -*- c++ -*-

// Local include(s).
#include "HelperClass.cuh"
#include "Macros.cuh"

// CUDA include(s).
#include <cuda.h>

// System include(s).
#include <cstring>

__host__
HelperClass::HelperClass( std::size_t csize )
   : m_csize( csize ) {

}

__host__ __device__
HelperClass::HelperClass( std::size_t csize, std::size_t vsize,
                          float** variables )
   : m_arrays( variables ), m_csize( csize ), m_vsize( vsize ) {

}

__host__ __device__
HelperClass::~HelperClass() {

#ifndef __CUDA_ARCH__
   // Note that I'm not using CUDA_CHECK(...) here, as the code would have no
   // way of handling the errors anyway...
   for( std::size_t i = 0; i < m_vsize; ++i ) {
      if( m_arrays[ i ] != nullptr ) {
         cudaFree( m_arrays[ i ] );
      }
   }
   if( m_arrays != nullptr ) {
      cudaFree( m_arrays );
   }
#endif // not __CUDA_ARCH__
}

__host__ __device__
std::size_t HelperClass::size() const {

   return m_csize;
}

__host__ __device__
const float* HelperClass::array( std::size_t id ) const {

   if( id >= m_vsize ) {
      return nullptr;
   }
   return m_arrays[ id ];
}
__host__ __device__
float* HelperClass::array( std::size_t id ) {

   if( id >= m_vsize ) {
      return nullptr;
   }
   return m_arrays[ id ];
}

__host__
void HelperClass::makeArray( std::size_t id ) {

   // Check if the internal structure is large enough.
   if( id >= m_vsize ) {
      // Decide about the size of the new array.
      std::size_t newVSize = m_vsize;
      if( newVSize == 0 ) {
         newVSize = 8;
      }
      while( newVSize <= id ) {
         newVSize *= 2;
      }
      // Allocate a new array.
      float** newArrays = nullptr;
      CUDA_CHECK( cudaMallocManaged( &newArrays,
                                     newVSize * sizeof( float* ) ) );
      // Copy the old array's payload to it.
      for( std::size_t i = 0; i < m_vsize; ++i ) {
         newArrays[ i ] = m_arrays[ i ];
      }
      for( std::size_t i = m_vsize; i < newVSize; ++i ) {
         newArrays[ i ] = nullptr;
      }
      // Delete the old array from memory.
      CUDA_CHECK( cudaFree( m_arrays ) );
      // Replace the old array with the new one.
      m_arrays = newArrays;
      m_vsize = newVSize;
   }

   // Check if anything needs to be done.
   if( m_arrays[ id ] != nullptr ) {
      return;
   }

   // Allocate a new managed array.
   CUDA_CHECK( cudaMallocManaged( &( m_arrays[ id ] ),
                                  m_csize * sizeof( float ) ) );
   // Make sure it's initialised to zero.
   std::memset( m_arrays[ id ], 0, m_csize );
   return;
}

__host__
std::pair< std::size_t, float** > HelperClass::arrays() const {

   return std::make_pair( m_vsize, m_arrays );
}
