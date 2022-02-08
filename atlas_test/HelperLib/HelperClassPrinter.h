// Dear emacs, this is -*- c++ -*-
#ifndef TEST11_HELPERCLASSPRINTER_H
#define TEST11_HELPERCLASSPRINTER_H

// System include(s).
#include <iosfwd>

// Forward declaration(s).
class HelperClass;

namespace std {

   /// Operator for printing the payload of an object of this type
   std::ostream& operator<<( std::ostream& out, const HelperClass& helper );

} // namespace std

#endif // TEST11_HELPERCLASSPRINTER_H
