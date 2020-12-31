#include "local_surface.h"
#include "utils.h"


torch::Tensor local_surface(torch::Tensor input,
                            torch::Tensor valid_mask,
                            int delta_t,
                            int r,
                            float tau){

    CHECK_INPUT(input);
    CHECK_INPUT(valid_mask);
    CHECK_IS_UINT8(valid_mask);

    return local_surface_wrapper(input, valid_mask, delta_t, r, tau);
}
