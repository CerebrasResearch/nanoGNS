#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <torch/extension.h>
#include <ATen/cuda/Atomic.cuh>

const uint32_t WARP_SIZE = 32;
const uint32_t N_THREADS = 1024;

template <typename T>
__device__ T warp_shuffle_sum(T sum) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    return sum;
}

template <typename T>
__global__ void layernorm_fwd_kernel(
    torch::PackedTensorAccessor64<T, 2, torch::RestrictPtrTraits> out,  
    torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> mean,
    torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> rstd,
    const torch::PackedTensorAccessor64<T, 2, torch::RestrictPtrTraits> in,
    const torch::PackedTensorAccessor64<T, 1, torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor64<T, 1, torch::RestrictPtrTraits> bias,
    const float eps
) {
    extern __shared__ float s_mem[];
    float* s_mean = s_mem;
    float* s_rstd = &s_mean[1];
    // max 32 warps
    float* warp_sum = &s_rstd[1];
    float* warp_sqsum = &warp_sum[N_THREADS / WARP_SIZE];
    T* s_in = (T*)(&warp_sqsum[N_THREADS / WARP_SIZE]);

    uint32_t b = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t warp_id = tid / WARP_SIZE;
    uint32_t lane_id = tid % WARP_SIZE;
    if(warp_id == 0) {
        warp_sum[lane_id] = 0.0;
        warp_sqsum[lane_id] = 0.0;
    }
    __syncthreads();

    uint32_t n = in.size(1);
    uint32_t elem_per_thread = (n + blockDim.x - 1) / blockDim.x;
    float local_sum = 0.0;
    for (uint32_t i = 0; i < elem_per_thread; i++) {
        uint32_t idx = tid + i * blockDim.x;
        if(idx >= n) break;
        T x = in[b][idx];
        local_sum += float(x);
        s_in[idx] = x;
    }

    local_sum = warp_shuffle_sum(local_sum);
    if(lane_id == 0) {
        warp_sum[warp_id] = local_sum;
    }
    __syncthreads();
    if(warp_id == 0) {
        local_sum = warp_sum[lane_id];
        local_sum = warp_shuffle_sum(local_sum);
        if(lane_id == 0) {
            *s_mean = local_sum / n;
        }
    }
    __syncthreads();
    float mean_val = *s_mean;

    float local_sqsum = 0.0;
    for (uint32_t i = 0; i < elem_per_thread; i++) {
        uint32_t idx = tid + i * blockDim.x;
        if(idx >= n) break;
        float x = float(s_in[idx]);
        float mcx = x - mean_val;
        local_sqsum += mcx * mcx;
    }
    local_sqsum = warp_shuffle_sum(local_sqsum);
    if(lane_id == 0) {
        warp_sqsum[warp_id] = local_sqsum;
    }
    __syncthreads();
    if(warp_id == 0) {
        local_sqsum = warp_sqsum[lane_id];
        local_sqsum = warp_shuffle_sum(local_sqsum);
        if(lane_id == 0) {
            *s_rstd = rsqrt((local_sqsum / n) + eps);
        }
    }
    __syncthreads();

    float rstd_val = *s_rstd;

    for(uint32_t i = 0; i < elem_per_thread; i++) {
        uint32_t idx = tid + i * blockDim.x;
        if(idx >= n) break;
        T x = s_in[idx];
        float y = (float(x) - mean_val) * rstd_val;
        T out_val = T(y) * weight[idx] + bias[idx];
        out[b][idx] = out_val;
    }

    if(tid == 0) {
        mean[b] = mean_val;
        rstd[b] = rstd_val;
    }
}

template <typename T>
__global__ void layernorm_bwd_kernel(
    torch::PackedTensorAccessor64<T, 3> grad_output,
    torch::PackedTensorAccessor64<float, 2> weight_grad_output,
    torch::PackedTensorAccessor64<float, 2> bias_grad_output,
    const torch::PackedTensorAccessor64<T, 3> grad_input,
    const torch::PackedTensorAccessor64<T, 3> in,
    const torch::PackedTensorAccessor64<T, 1> weight,
    const torch::PackedTensorAccessor64<float, 2> mean,
    const torch::PackedTensorAccessor64<float, 2> rstd
) {
    uint32_t b = blockIdx.x;
    uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t seq_len = in.size(1);
    uint32_t n = in.size(2);
    uint32_t elem_per_thread = (n + blockDim.x - 1) / blockDim.x;

    extern __shared__ float s_mem[];
    float* shared_dnorm_sum = s_mem;
    float* shared_dnorm_norm_sum = &shared_dnorm_sum[blockDim.x];
    float* shared_w_grad = &shared_dnorm_norm_sum[blockDim.y];
    float* shared_b_grad = &shared_w_grad[n];
    
    for(uint32_t i = 0; i < elem_per_thread; i++) {
        uint32_t idx = threadIdx.x + i * blockDim.x;
        if(idx >= n) break;
        if(idx < blockDim.y) {
            shared_dnorm_sum[idx] = 0.0;
            shared_dnorm_norm_sum[idx] = 0.0;
        }
        shared_w_grad[idx] = 0.0;
        shared_b_grad[idx] = 0.0;
    }
    __syncthreads();

    T mean_val = seq_idx < seq_len ? mean[b][seq_idx] : 0.0;
    T rstd_val = seq_idx < seq_len ? rstd[b][seq_idx] : 0.0;

    float local_dnorm_sum = 0.0;
    float local_dnorm_norm_sum = 0.0;
    for(uint32_t i = 0; i < elem_per_thread; i++) {
        uint32_t idx = threadIdx.x + i * blockDim.x;
        if(idx >= n) break;
        if(seq_idx >= seq_len) {
            continue;
        }
        float grad = float(grad_input[b][seq_idx][idx]);
        float x = float(in[b][seq_idx][idx]);
        float x_norm = (x - mean_val) * rstd_val;
        float weight_grad = grad * x_norm;
        atomicAdd(&shared_w_grad[idx], weight_grad);
        atomicAdd(&shared_b_grad[idx], grad);
        
        float w = float(weight[idx]);
        float dnorm = grad * w;

        local_dnorm_sum += dnorm;
        local_dnorm_norm_sum += dnorm * x_norm;
    }
    __syncthreads();

    local_dnorm_sum = warp_shuffle_sum(local_dnorm_sum);
    local_dnorm_norm_sum = warp_shuffle_sum(local_dnorm_norm_sum);

    if((threadIdx.x % 32) == 0) {
        atomicAdd(&shared_dnorm_sum[threadIdx.y], local_dnorm_sum);
        atomicAdd(&shared_dnorm_norm_sum[threadIdx.y], local_dnorm_norm_sum);
    }
    __syncthreads();

    float dnorm_mean = shared_dnorm_sum[threadIdx.y] / n;
    float dnorm_norm_mean = shared_dnorm_norm_sum[threadIdx.y] / n;

    for(uint32_t i = 0; i < elem_per_thread; i++) {
        uint32_t idx = threadIdx.x + i * blockDim.x;
        if(idx >= n || seq_idx >= seq_len) break;

        float grad = float(grad_input[b][seq_idx][idx]);
        float w = float(weight[idx]);
        float dnorm = grad * w;
        float x_norm = (float(in[b][seq_idx][idx]) - mean_val) * rstd_val;

        grad_output[b][seq_idx][idx] = T((dnorm - dnorm_mean - x_norm * dnorm_norm_mean) * rstd_val);
        if(threadIdx.y == 0) {
            atomicAdd(&weight_grad_output[b][idx], shared_w_grad[idx]);
            atomicAdd(&bias_grad_output[b][idx], shared_b_grad[idx]);
        }
    }
}

std::vector<torch::Tensor> layernorm_fwd(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    auto out = torch::empty_like(x);
    auto mean = torch::empty({x.size(0)}, x.options().dtype(torch::kFloat));
    auto rstd = torch::empty({x.size(0)}, x.options().dtype(torch::kFloat));
    const dim3 blocks(x.size(0));
    const int threads = N_THREADS;
    int shared_size = (2 + 2 * (N_THREADS / WARP_SIZE)) * sizeof(float);
    switch (x.scalar_type()) {
        case torch::ScalarType::Double:
            shared_size += x.size(1) * sizeof(double);
            layernorm_fwd_kernel<double><<<blocks, threads, shared_size>>>(
                out.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                mean.packed_accessor64<float, 1, torch::RestrictPtrTraits>(),
                rstd.packed_accessor64<float, 1, torch::RestrictPtrTraits>(),
                x.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                weight.packed_accessor64<double, 1, torch::RestrictPtrTraits>(),
                bias.packed_accessor64<double, 1, torch::RestrictPtrTraits>(),
                eps
            );
            break;
        case torch::ScalarType::Float:
            shared_size += x.size(1) * sizeof(float);
            layernorm_fwd_kernel<float><<<blocks, threads, shared_size>>>(
                out.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                mean.packed_accessor64<float, 1, torch::RestrictPtrTraits>(),
                rstd.packed_accessor64<float, 1, torch::RestrictPtrTraits>(),
                x.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                weight.packed_accessor64<float, 1, torch::RestrictPtrTraits>(),
                bias.packed_accessor64<float, 1, torch::RestrictPtrTraits>(),
                eps
            );
            break;
        case torch::ScalarType::Half:
            shared_size += x.size(1) * sizeof(at::Half);
            layernorm_fwd_kernel<at::Half><<<blocks, threads, shared_size>>>(
                out.packed_accessor64<at::Half, 2, torch::RestrictPtrTraits>(),
                mean.packed_accessor64<float, 1, torch::RestrictPtrTraits>(),
                rstd.packed_accessor64<float, 1, torch::RestrictPtrTraits>(),
                x.packed_accessor64<at::Half, 2, torch::RestrictPtrTraits>(),
                weight.packed_accessor64<at::Half, 1, torch::RestrictPtrTraits>(),
                bias.packed_accessor64<at::Half, 1, torch::RestrictPtrTraits>(),
                eps
            );
            break;
        case torch::ScalarType::BFloat16:
            shared_size += x.size(1) * sizeof(at::BFloat16);
            layernorm_fwd_kernel<at::BFloat16><<<blocks, threads, shared_size>>>(
                out.packed_accessor64<at::BFloat16, 2, torch::RestrictPtrTraits>(),
                mean.packed_accessor64<float, 1, torch::RestrictPtrTraits>(),
                rstd.packed_accessor64<float, 1, torch::RestrictPtrTraits>(),
                x.packed_accessor64<at::BFloat16, 2, torch::RestrictPtrTraits>(),
                weight.packed_accessor64<at::BFloat16, 1, torch::RestrictPtrTraits>(),
                bias.packed_accessor64<at::BFloat16, 1, torch::RestrictPtrTraits>(),
                eps
            );
            break;    
        default:
            TORCH_CHECK(false, "Unsupported dtype");
            break;
    }
    return {out, mean, rstd};
}

std::vector<torch::Tensor> layernorm_bwd(
    torch::Tensor grad_input,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor mean,
    torch::Tensor rstd
) {
    TORCH_CHECK(grad_input.is_cuda(), "grad_input must be a CUDA tensor");
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(mean.is_cuda(), "mean must be a CUDA tensor");
    TORCH_CHECK(rstd.is_cuda(), "rstd must be a CUDA tensor");
    TORCH_CHECK(grad_input.dim() == 3, "grad_input must be 3D");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(mean.dim() == 2, "mean must be 2D");
    TORCH_CHECK(rstd.dim() == 2, "rstd must be 2D");
    TORCH_CHECK(grad_input.size(0) == x.size(0), "batch size mismatch");
    TORCH_CHECK(grad_input.size(1) == x.size(1), "sequence length mismatch");
    TORCH_CHECK(grad_input.size(2) == x.size(2), "hidden size mismatch");
    TORCH_CHECK(mean.size(0) == x.size(0), "batch size mismatch");
    TORCH_CHECK(mean.size(1) == x.size(1), "sequence length mismatch");
    TORCH_CHECK(rstd.size(0) == x.size(0), "batch size mismatch");
    TORCH_CHECK(rstd.size(1) == x.size(1), "sequence length mismatch");
    auto grad_output = torch::zeros_like(grad_input);
    auto weight_grad_output = torch::zeros({x.size(0), x.size(2)}, x.options().dtype(torch::kFloat));
    auto bias_grad_output = torch::zeros({x.size(0), x.size(2)}, x.options().dtype(torch::kFloat));
    const dim3 blocks(x.size(0), (x.size(1) + 32 - 1) / 32, 1);
    const dim3 threads(N_THREADS / 32, 32, 1);
    const int shared_size = (2 * 32 + 2 * x.size(2)) * sizeof(float);
    switch (x.scalar_type()) {
        case torch::ScalarType::Double:
            layernorm_bwd_kernel<double><<<blocks, threads, shared_size>>>(
                grad_output.packed_accessor64<double, 3>(),
                weight_grad_output.packed_accessor64<float, 2>(),
                bias_grad_output.packed_accessor64<float, 2>(),
                grad_input.packed_accessor64<double, 3>(),
                x.packed_accessor64<double, 3>(),
                weight.packed_accessor64<double, 1>(),
                mean.packed_accessor64<float, 2>(),
                rstd.packed_accessor64<float, 2>()
            );
            break;
        case torch::ScalarType::Float:
            layernorm_bwd_kernel<float><<<blocks, threads, shared_size>>>(
                grad_output.packed_accessor64<float, 3>(),
                weight_grad_output.packed_accessor64<float, 2>(),
                bias_grad_output.packed_accessor64<float, 2>(),
                grad_input.packed_accessor64<float, 3>(),
                x.packed_accessor64<float, 3>(),
                weight.packed_accessor64<float, 1>(),
                mean.packed_accessor64<float, 2>(),
                rstd.packed_accessor64<float, 2>()
            );
            break;
        case torch::ScalarType::Half:
            layernorm_bwd_kernel<at::Half><<<blocks, threads, shared_size>>>(
                grad_output.packed_accessor64<at::Half, 3>(),
                weight_grad_output.packed_accessor64<float, 2>(),
                bias_grad_output.packed_accessor64<float, 2>(),
                grad_input.packed_accessor64<at::Half, 3>(),
                x.packed_accessor64<at::Half, 3>(),
                weight.packed_accessor64<at::Half, 1>(),
                mean.packed_accessor64<float, 2>(),
                rstd.packed_accessor64<float, 2>()
            );
            break;
        case torch::ScalarType::BFloat16:
            layernorm_bwd_kernel<at::BFloat16><<<blocks, threads, shared_size>>>(
                grad_output.packed_accessor64<at::BFloat16, 3>(),
                weight_grad_output.packed_accessor64<float, 2>(),
                bias_grad_output.packed_accessor64<float, 2>(),
                grad_input.packed_accessor64<at::BFloat16, 3>(),
                x.packed_accessor64<at::BFloat16, 3>(),
                weight.packed_accessor64<at::BFloat16, 1>(),
                mean.packed_accessor64<float, 2>(),
                rstd.packed_accessor64<float, 2>()
            );
            break;
        default:
            TORCH_CHECK(false, "Unsupported dtype");
            break;
    }
    return {grad_output, weight_grad_output, bias_grad_output};
}
