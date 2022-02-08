// Dear emacs, this is -*- c++ -*-

// Local include(s).
#include "HelperClass.cuh"

#if defined(_WIN32) || defined(_WIN64)
__declspec(dllexport)
#endif
HelperClass*
#if defined(_WIN32) || defined(_WIN64)
__cdecl
#endif
prepareObject() {

   // Construct the helper object.
   static const std::size_t SIZE = 10;
   HelperClass* helper = new HelperClass( SIZE );
   helper->makeArray( 0 );
   helper->makeArray( 3 );
   for( std::size_t i = 0; i < SIZE; ++i ) {
      helper->array( 0 )[ i ] = i * 0.5f;
      helper->array( 3 )[ i ] = i * 1.5f;
   }

   // Return it to the user.
   return helper;
}
