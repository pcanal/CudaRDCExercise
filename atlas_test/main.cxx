
// Local include(s).
#include "PrepareObject.h"
#include "HelperClassPrinter.h"
#include "ArrayMultiply.h"

// System include(s).
#include <iostream>

int main() {

   // Construct the helper object.
   HelperClass* helper = prepareObject();
   std::cout << "Original:\n" << *helper << std::endl;

   // Perform "the" operation on the object.
   arrayMultiply( *helper );

   // Check what happened to it.
   std::cout << "Processed:\n" << *helper << std::endl;

   // Return gracefully.
   return 0;
}
