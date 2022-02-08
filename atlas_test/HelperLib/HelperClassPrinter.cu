// Dear emacs, this is -*- c++ -*-

// Local include(s).
#include "HelperClassPrinter.h"
#include "HelperClass.cuh"

// System include(s).
#include <iostream>

namespace {

   void print( std::ostream& out, std::size_t size, const float* array ) {

      out << "[";
      if( array ) {
         for( std::size_t i = 0; i < size; ++i ) {
            out << array[ i ];
            if( i + 1 < size ) {
               out << ", ";
            }
         }
      }
      out << "]";
   }

} // private namespace

namespace std {

   std::ostream& operator<<( std::ostream& out, const HelperClass& helper ) {

      auto arrays = helper.arrays();
      for( std::size_t i = 0; i < arrays.first; ++i ) {
         ::print( out, helper.size(), helper.array( i ) );
         if( i + 1 < arrays.first ) {
            out << std::endl;
         }
      }
      return out;
   }

} // namespace std
