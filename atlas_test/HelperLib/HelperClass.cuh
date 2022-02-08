// Dear emacs, this is -*- c++ -*-
#ifndef TEST11_HELPERCLASS_H
#define TEST11_HELPERCLASS_H

// System include(s).
#include <cstddef>
#include <utility>

/// Dummy class helping with the CUDA code
class HelperClass {

public:
   /// Default constructor
   __host__
   HelperClass( std::size_t csize );
   /// Constructor from an existing set of variables
   __host__ __device__
   HelperClass( std::size_t csize, std::size_t vsize,
                float** variables );
   /// Destructor
   __host__ __device__
   ~HelperClass();

   /// @name Function(s) for both the host and the device
   /// @{

   /// Get the size of the arrays held by the container
   __host__ __device__
   std::size_t size() const;
   /// Access one specific array (const)
   __host__ __device__
   const float* array( std::size_t id ) const;
   /// Access one specific array (non-const)
   __host__ __device__
   float* array( std::size_t id );

   /// @}

   /// @name Function(s) for just the host
   /// @{

   /// Create an array with a given ID
   __host__
   void makeArray( std::size_t id );

   /// Access the internal variables, to be sent to device code
   __host__
   std::pair< std::size_t, float** > arrays() const;

   /// @}

private:
   /// Pointers to the managed arrays
   float** m_arrays = nullptr;
   /// Size of the arrays
   std::size_t m_csize = 0;
   /// Number of managed arrays
   std::size_t m_vsize = 0;

}; // class HelperClass

#endif // TEST11_HELPERCLASS_H
