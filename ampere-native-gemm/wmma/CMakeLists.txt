add_executable(ampere_wmma driver.cu)
target_link_libraries(ampere_wmma PRIVATE nt_common)
nt_set_native_archs(ampere_wmma)
target_compile_options(ampere_wmma PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -lineinfo>)
