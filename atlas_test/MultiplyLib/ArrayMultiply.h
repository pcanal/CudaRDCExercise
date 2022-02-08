// Dear emacs, this is -*- c++ -*-
#ifndef TEST11_ARRAYMULTIPLY_H
#define TEST11_ARRAYMULTIPLY_H

// Forward declaration(s).
class HelperClass;

/// Function multiplying array with index 0 in the helper object
#if defined(_WIN32) || defined(_WIN64)
__declspec(dllexport)
#endif
void
#if defined(_WIN32) || defined(_WIN64)
__cdecl
#endif
arrayMultiply( HelperClass& helper );

#endif // TEST11_ARRAYMULTIPLY_H
