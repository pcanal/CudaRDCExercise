// Dear emacs, this is -*- c++ -*-
#ifndef TEST11_MACROS_CUH
#define TEST11_MACROS_CUH

// System include(s).
#include <stdexcept>
#include <string>

/// Helper macro for checking CUDA function return codes
#define CUDA_CHECK( EXP )                                            \
   do {                                                              \
      if( EXP != cudaSuccess ) {                                     \
         throw std::runtime_error( std::string( __FILE__ ) + ":" +   \
                                   std::to_string( __LINE__ ) +      \
                                   " Failed to execute: " #EXP );    \
      }                                                              \
   } while( false )

#endif // TEST11_MACROS_CUH
