set(__SRCS__)
set(__INCLUDE_DIRS__)
set(__LINK_LIBS__)

file(GLOB __TMP_SRCS__
    *.cu)
list(APPEND __SRCS__ ${__TMP_SRCS__})
unset(__TMP_SRCS__)

list(APPEND __INCLUDE_DIRS__ ${CUDAToolkit_INCLUDE_DIRS})

list(APPEND __LINK_LIBS__
    CUDA::cuda_driver CUDA::cudart CUDA::cublasLt CUDA::cupti)

get_property(THRUST_TARGET GLOBAL PROPERTY GLOBAL_THRUST_TARGET)
if(THRUST_TARGET)
    list(APPEND __LINK_LIBS__ ${THRUST_TARGET})
endif()

# add_executable(gemm ${__SRCS__})
# set_property(TARGET gemm PROPERTY CUDA_ARCHITECTURES 90a)

add_executable(gemm_self self.cu)
set_property(TARGET gemm_self PROPERTY CUDA_ARCHITECTURES 90a)

add_executable(gemm gemm.cu)
set_property(TARGET gemm PROPERTY CUDA_ARCHITECTURES 90a)

target_include_directories(gemm PUBLIC
    ${__INCLUDE_DIRS__})

target_link_libraries(gemm PRIVATE
    ${__LINK_LIBS__})

set_property(TARGET gemm PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

target_compile_options(gemm PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--gpu-architecture=sm_90a>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

unset(__SRCS__)
unset(__INCLUDE_DIRS__)
unset(__LINK_LIBS__)