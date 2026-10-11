add_library(nt_common INTERFACE)
target_include_directories(nt_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(nt_common INTERFACE CUDA::cublas CUDA::cudart)
