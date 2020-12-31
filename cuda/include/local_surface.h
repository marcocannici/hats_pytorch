#pragma once
#include <torch/extension.h>

torch::Tensor local_surface(torch::Tensor input,
                            torch::Tensor valid_mask,
                            int delta_t,
                            int r,
                            float tau);


torch::Tensor local_surface_wrapper(torch::Tensor input,
                                    torch::Tensor valid_mask,
                                    int delta_t,
                                    int r,
                                    float tau);