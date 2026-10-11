# Architectures this dev box can both COMPILE and RUN.
# 80  = Ampere (A100)          -- common cp.async / mma.sync target
# 89  = Ada
# 90  = Hopper *without* the 'a' suffix -- portable/forward-compatible codegen,
#       does NOT unlock wgmma/TMA specific instructions (those need 90a)
# 120 = Blackwell, RTX 5070 Ti -- this dev machine, can actually execute
set(NATIVE_ARCHS "80;89;90;120" CACHE STRING "Architectures built AND runnable locally")

# Hopper-native ('a' variant) unlocks wgmma.mma_async / cp.async.bulk.tensor / mbarrier
# ISA that plain sm_90 codegen does not expose. This machine has no Hopper GPU, so any
# target pinned to 90a is COMPILE-ONLY here (ptxas can still assemble it) -- execution
# and profiling happen later on a rented H100.
set(HOPPER_ARCH "90a" CACHE STRING "Hopper-native arch, compile-only on this machine")

function(nt_set_native_archs target)
    set_property(TARGET ${target} PROPERTY CUDA_ARCHITECTURES "${NATIVE_ARCHS}")
endfunction()

function(nt_set_hopper_arch target)
    set_property(TARGET ${target} PROPERTY CUDA_ARCHITECTURES "${HOPPER_ARCH}")
endfunction()
