#include "util.hh"
#include "std_util.hh"

#ifndef TORCH_INC_H_                    // Prevent torch
#define TORCH_INC_H_                    // from reloading
#include <torch/torch.h>                //
#include <torch/nn/functional/linear.h> //
#endif                                  // End of torch

#ifndef MMVAE_MODULE_LINEARL2_HH_
#define MMVAE_MODULE_LINEARL2_HH_

namespace mmvae {

struct LinearL2Impl : public torch::nn::Module {

    explicit LinearL2Impl(int64_t in_features, int64_t out_features);

    void reset();

    void reset_parameters();

    /// Transforms the `input` tensor by multiplying with the `weight` and
    /// optionally adding the `bias`, if `with_bias` is true in the options.
    torch::Tensor forward(const torch::Tensor &input);

    torch::Tensor weight; // unnormalized weights
    torch::Tensor bias;

    const int64_t d_in;
    const int64_t d_out;
    const float penalty;
};

torch::Tensor
LinearL2Impl::forward(const torch::Tensor &input)
{
    return F::linear(input, weight, bias);
}

LinearL2Impl::LinearL2Impl(int64_t in_features, int64_t out_features)
    : d_in(in_features)
    , d_out(out_features)
    , penalty(1e-2)
{
    reset();
}

void
LinearL2Impl::reset()
{
    weight = register_parameter("weight", torch::empty({ d_out, d_in }));
    bias = register_parameter("bias", {}, false);
    reset_parameters();
}

void
LinearL2Impl::reset_parameters()
{
    torch::nn::init::kaiming_uniform_(weight, std::sqrt(5));
    if (bias.defined()) {
        int64_t fan_in, fan_out;
        std::tie(fan_in, fan_out) =
            torch::nn::init::_calculate_fan_in_and_fan_out(weight);
        const auto bound = 1 / std::sqrt(fan_in);
        torch::nn::init::uniform_(bias, -bound, bound);
    }
}

TORCH_MODULE(LinearL2);
}
#endif
