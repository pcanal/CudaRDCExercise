# Testing that we can produce MODULE libraries

# 1a. Build "mod_foo" as a full, shared module RDC library
cuda_rdc_add_library(mod_foo SHARED foo.h foo.cu)

# 1b. Build "uses_foo" as a full shared module without CUDA which depends on a shared RDC library
cuda_rdc_add_library(uses_foo MODULE usesfoo.cc)
cuda_rdc_target_link_libraries(uses_foo mod_foo)

# 1c. Build "uses_foo" as a full shared module with CUDA which depends on a shared RDC library
cuda_rdc_add_library(uses_cuda_foo MODULE usesfoo.cc dummy.cu)
cuda_rdc_target_link_libraries(uses_cuda_foo mod_foo)

# 1d. Consume "mod_foo_final" in a pure C++ exe
add_executable(mod_foo_CPP_Consumer fooConsumer.cc)
cuda_rdc_target_link_libraries(mod_foo_CPP_Consumer ${CMAKE_DL_LIBS})
add_dependencies(mod_foo_CPP_Consumer mod_foo)
add_test(NAME test_mod_foo_CPP_Consumer COMMAND mod_foo_CPP_Consumer)

# 1e. Consume "mod_foo_final" in a mixed C++/Cuda exe
add_executable(mod_foo_CUDA_Consumer fooConsumer.cc dummy.cu)
cuda_rdc_target_link_libraries(mod_foo_CUDA_Consumer ${CMAKE_DL_LIBS})
set_target_properties(mod_foo_CUDA_Consumer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_dependencies(mod_foo_CUDA_Consumer mod_foo)
add_test(NAME test_mod_foo_CUDA_Consumer COMMAND mod_foo_CUDA_Consumer)

# 1f. Consume "uses_foo" in a pure C++ exe
add_executable(uses_foo_CPP_Consumer usesfooConsumer.cc)
cuda_rdc_target_link_libraries(uses_foo_CPP_Consumer ${CMAKE_DL_LIBS})
add_dependencies(uses_foo_CPP_Consumer uses_foo)
add_test(NAME test_uses_foo_CPP_Consumer COMMAND uses_foo_CPP_Consumer)

# 1g. Consume "uses_cuda_foo" in a pure C++ exe
add_executable(uses_cuda_foo_CPP_Consumer usesCudafooConsumer.cc)
cuda_rdc_target_link_libraries(uses_cuda_foo_CPP_Consumer ${CMAKE_DL_LIBS})
add_dependencies(uses_cuda_foo_CPP_Consumer uses_foo)
add_test(NAME test_uses_cuda_foo_CPP_Consumer COMMAND uses_cuda_foo_CPP_Consumer)
