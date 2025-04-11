#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

int main(int argc, char** argv) {
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS init failed\n");
        return -1;
    }

    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    size_t A_size = M * K;
    size_t B_size = K * N;
    size_t C_size = M * N;
    
    half *h_A = (half*)malloc(A_size * sizeof(half));
    half *h_B = (half*)malloc(B_size * sizeof(half));
    half *h_C = (half*)malloc(C_size * sizeof(half));
    
    if (!h_A || !h_B || !h_C) {
        printf("memory alloc failed\n");
        return -1;
    }
    
    // 随机初始化矩阵
    for (int i = 0; i < A_size; i++) {
        h_A[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }
    
    for (int i = 0; i < B_size; i++) {
        h_B[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }
    
    half *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, A_size * sizeof(half));
    cudaMalloc((void**)&d_B, B_size * sizeof(half));
    cudaMalloc((void**)&d_C, C_size * sizeof(half));

    cudaMemcpy(d_A, h_A, A_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, B_size * sizeof(half), cudaMemcpyHostToDevice);
    
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);
    
    for (int i = 0; i < 10; i++) {
        status = cublasGemmEx(handle,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              N, M, K,
                              &alpha,
                              d_B, CUDA_R_16F, K,
                              d_A, CUDA_R_16F, K,
                              &beta,
                              d_C, CUDA_R_16F, M,
                              CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("cuBLAS HGEMM failed during warmup: %d\n", status);
            return -1;
        }
    }
    
    cudaStreamSynchronize(stream);
    
    const int num_iters = 20;
    float total_ms = 0.0f;
    float min_ms = 1e10f;
    float max_ms = 0.0f;

    for (int iter = 0; iter < num_iters; iter++) {
        cudaEventRecord(start, stream);
        
        status = cublasGemmEx(handle,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              N, M, K,
                              &alpha,
                              d_B, CUDA_R_16F, K,
                              d_A, CUDA_R_16F, K,
                              &beta,
                              d_C, CUDA_R_16F, M,
                              CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("cuBLAS HGEMM failed: %d\n", status);
            return -1;
        }
        
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        
        total_ms += ms;
        min_ms = min(min_ms, ms);
        max_ms = max(max_ms, ms);
    }
    
    float avg_ms = total_ms / num_iters;
    double flops = 2.0 * M * N * K;
    double avg_gflops = (flops * 1.0e-9) / (avg_ms * 1.0e-3);
    double peak_gflops = (flops * 1.0e-9) / (min_ms * 1.0e-3);
    
    printf("BLAS_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", avg_gflops, avg_ms);

    cudaMemcpy(h_C, d_C, C_size * sizeof(half), cudaMemcpyDeviceToHost);
    
    cudaStreamDestroy(stream);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    
    return 0;
}