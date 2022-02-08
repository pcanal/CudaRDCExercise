// Dear emacs, this is -*- c++ -*-
#ifndef TEST11_PREPAREOBJECT_H
#define TEST11_PREPAREOBJECT_H

// Forward declaration(s).
class HelperClass;

/// Function creating an instance of HelperClass
#if defined(_WIN32) || defined(_WIN64)
__declspec(dllexport)
#endif
HelperClass*
#if defined(_WIN32) || defined(_WIN64)
__cdecl
#endif
prepareObject();

#endif // TEST11_PREPAREOBJECT_H
