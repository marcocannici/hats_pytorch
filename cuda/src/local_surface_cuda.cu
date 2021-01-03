#include <torch/extension.h>
#include <vector>
#include <stdio.h>

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>


template <typename scalar_t>
__global__ void local_surface_kernel(const scalar_t* __restrict__ input,
                                     const int64_t* __restrict__ lengths,
                                     float* __restrict__ output,
			                         const int64_t event_size,
                                     const int64_t batch_size,
			                         const int64_t feature_size,
                                     const double delta_t,
                                     const int64_t r,
                                     const double tau){

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    const int rf_size = 2 * r + 1;

    if (i < event_size & b < batch_size){
        int64_t i_idx = i * batch_size * feature_size + b * feature_size;
        int i_x = input[i_idx + 0];
        int i_y = input[i_idx + 1];
        auto i_t = input[i_idx + 2];
        int i_p = input[i_idx + 3];

        if (i < lengths[b]){
            // Look into the past memory
            for(int64_t e = i - 1; e >= 0; e--){
                int64_t e_idx = e * batch_size * feature_size + b * feature_size;
                int e_x = input[e_idx + 0];
                int e_y = input[e_idx + 1];
                auto e_t = input[e_idx + 2];
                int e_p = input[e_idx + 3];

                // We stop as soon as we exit the event's memory
                if (e_t < i_t - delta_t)
                    break;

                int rf_y = e_y - i_y + r;
                int rf_x = e_x - i_x + r;

                // The eth event is inside the eih event's neighborhood
                if ((e_t < i_t) & (e_p == i_p) &
                    (rf_x < rf_size) & (rf_y < rf_size)){

                    float value = expf((float)(e_t - i_t) / tau);
                    int64_t out_idx = i * 2 * rf_size * rf_size * batch_size + \
                                      i_p * rf_size * rf_size * batch_size + \
                                      rf_y * rf_size * batch_size + \
                                      rf_x * batch_size + \
                                      b;
                    output[out_idx] = output[out_idx] + value;
                }
            }
        }
    }
}


torch::Tensor local_surface_wrapper(torch::Tensor input,
                                    torch::Tensor lengths,
                                    double delta_t,
                                    int r,
                                    double tau){

    const auto event_size = input.size(0);
    const auto batch_size = input.size(1);
    const auto feature_size = input.size(2);

    // Create a tensor to hold the result
	auto output = torch::zeros({event_size, 2, 2*r + 1, 2*r + 1, batch_size},
	                            input.options().dtype(at::kFloat));

    // Split the first dimension over threadsPerBlock.x threads
    // and the second dimension over threadsPerBlock.y threads
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((int)((event_size + threadsPerBlock.x - 1) / threadsPerBlock.x),
                   (int)((batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y));

    AT_DISPATCH_ALL_TYPES(input.type(), "local_surface_wrapper", ([&] {
		local_surface_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
			input.data_ptr<scalar_t>(),
			lengths.data_ptr<int64_t>(),
			output.data_ptr<float>(),
			event_size,
			batch_size,
			feature_size,
			delta_t,
			r,
			tau);
	}));

    return output;
}