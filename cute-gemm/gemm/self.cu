#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "cute/tensor.hpp"
#include "cute/int_tuple.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/underscore.hpp"


#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/device_kernel.h"


using namespace cute;

// shared memory struct
template <class ElementA,
          class ElementB,
          class SmemLayoutA,
          class SmemLayoutB>
struct SharedStorage
{
  alignas(128) ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  alignas(128) ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;

  uint64_t tma_barrier[size<2>(SmemLayoutA{})];
  uint64_t mma_barrier[size<2>(SmemLayoutA{})];
};

template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
            TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
            TC      * C, CStride dC, TiledMma mma,
            Alpha alpha, Beta beta)
{
    auto [M, N, K] = shape_MNK;
    Tensor mA = tma_a.get_tma_tensor(make_shape(M,K));
    Tensor mB = tma_b.get_tma_tensor(make_shape(N,K));
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC);

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});

    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});

    auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
        group_modes<0,2>(sA), group_modes<0,2>(gA));

    auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
            group_modes<0,2>(sB), group_modes<0,2>(gB));
    
    constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA)))
            + sizeof(make_tensor_like(tensor<0>(tBsB)));

    auto K_PIPE_MAX = size<1>(tAsA);
    int k_tile_count = size<1>(tAgA);
    int k_tile = 0;

    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();
    uint64_t* producer_mbar = smem.tma_barrier;
    uint64_t* consumer_mbar = smem.mma_barrier;
    
    using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBarType = cutlass::arch::ClusterBarrier;
    CUTE_UNROLL
    for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
      if ((warp_idx == 0) && lane_predicate) {
        ProducerBarType::init(&producer_mbar[pipe],   1);
        ConsumerBarType::init(&consumer_mbar[pipe], 128);
      }
    }
    cluster_sync();

    CUTE_UNROLL
    for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe)
    {
      if ((warp_idx == 0) && lane_predicate)
      {
        // Set expected Tx Bytes after each reset / init
        ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
        copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
        copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
      }
      --k_tile_count;
      ++k_tile;
    }
  
    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);

    auto write_state = cutlass::PipelineState<K_PIPE_MAX>();
    auto read_state = cutlass::PipelineState<K_PIPE_MAX>();

    CUTE_NO_UNROLL
    while (k_tile_count > -K_PIPE_MAX)
    {
        int read_pipe = read_state.index();
        ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());
        warpgroup_arrive();
        gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
        warpgroup_commit_batch();

        warpgroup_wait<0>();
        
        ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
        ++read_state;

        if ((warp_idx == 0) && lane_predicate)
        {
          int pipe = write_state.index();
          ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
          ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
          copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
          copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
          ++write_state;
        }
        --k_tile_count;
        ++k_tile;
    }
    axpby(alpha, tCrC, beta, tCgC);
}

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm(int m, int n, int k,
    Alpha alpha,
    TA const* A, int ldA,
    TB const* B, int ldB,
    Beta beta,
    TC      * C, int ldC,
    cudaStream_t stream = 0)
{
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);

    auto dA = make_stride(ldA, Int<1>{});
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(Int<1>{}, ldC);

    auto bM = Int<128>{};
    auto bN = Int<256>{};
    auto bK = Int< 64>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    auto bP = Int<3>{};

    auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
    auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

    TiledMMA tiled_mma = make_tiled_mma(SM90_64x256x16_F16F16F16_SS<GMMA::Major::K,GMMA::Major::K>{});
    Tensor mA = make_tensor(A, make_shape(M,K), dA);
    Tensor mB = make_tensor(B, make_shape(N,K), dB);
    Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
    Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));
  
    dim3 dimBlock(size(tiled_mma));
    dim3 dimCluster(2, 1, 1);
    dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
                round_up(size(ceil_div(n, bN)), dimCluster.y));
    int  smemBytes = sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>);

    auto* kernel_ptr = &gemm_device<decltype(prob_shape), decltype(cta_tiler),
                                    TA, decltype(sA), decltype(tmaA),
                                    TB, decltype(sB), decltype(tmaB),
                                    TC, decltype(dC), decltype(tiled_mma),
                                    decltype(alpha), decltype(beta)>;

    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            smemBytes));

    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smemBytes};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*) kernel_ptr,
                                                                prob_shape, cta_tiler,
                                                                A, tmaA,
                                                                B, tmaB,
                                                                C, dC, tiled_mma,
                                                                alpha, beta);
    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Error: Failed at kernel Launch" << std::endl;
    }

  
}

int main(int argc, char** argv)
{
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    const int lda = K;
    const int ldb = K;
    const int ldc = M;
    
    using TA = half_t;
    using TB = half_t;
    using TC = half_t;
    using TI = half_t;

    TI alpha = TI(1.0f);
    TI beta  = TI(0.0f);

    thrust::host_vector<TA> h_A(M * K);
    thrust::host_vector<TB> h_B(N * K);
    thrust::host_vector<TC> h_C(M * N);
  
    // Initialize the tensors
    for (int j = 0; j < M * K; ++j) h_A[j] = TA(int((rand() % 2) ? 1 : -1));
    for (int j = 0; j < N * K; ++j) h_B[j] = TB(int((rand() % 2) ? 1 : -1));
    for (int j = 0; j < M * N; ++j) h_C[j] = TC(0);
    
    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;
    
    double gflops = (2.0*M*N*K) * 1e-9;

    const int timing_iterations = 100;
    GPU_Clock timer;

    // Warm up
    gemm(M, N, K, 
         alpha, 
         d_A.data().get(), lda,
         d_B.data().get(), ldb, 
         beta,
         d_C.data().get(), ldc);
    CUTE_CHECK_LAST();
    thrust::host_vector<TC> cute_result = d_C;
  
    // Time iterations
    timer.start();
    for (int i = 0; i < timing_iterations; ++i) {
        gemm(M, N, K, 
             alpha, 
             d_A.data().get(), lda,
             d_B.data().get(), ldb, 
             beta,
             d_C.data().get(), ldc);
        CUTE_CHECK_LAST();
    }
    double cute_time = timer.seconds() / timing_iterations;
    CUTE_CHECK_LAST();
    printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);

    
    return 0;
}





