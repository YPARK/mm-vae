#include "util.hh"
#include "std_util.hh"

#ifndef TORCH_INC_H_                    // Prevent torch
#define TORCH_INC_H_                    // from reloading
#include <torch/torch.h>                //
#include <torch/nn/functional/linear.h> //
#endif                                  // End of torch

#ifndef MMVAE_MODULE_ANGULAR_HH_
#define MMVAE_MODULE_ANGULAR_HH_

namespace mmvae {

struct AngularImpl : public torch::nn::Module {

    explicit AngularImpl(int64_t in_features, int64_t out_features);

    void reset();

    void reset_parameters();

    /// Transforms the `input` tensor by multiplying with the `weight` and
    /// optionally adding the `bias`, if `with_bias` is true in the options.
    torch::Tensor forward(const torch::Tensor &input);

    torch::Tensor weight; // unnormalized weights
    torch::Tensor bias;

    const int64_t d_in;
    const int64_t d_out;
};

torch::Tensor
AngularImpl::forward(const torch::Tensor &input)
{
    const float eps = 1e-4;
    namespace F = torch::nn::functional;
    auto ww = F::normalize(weight + eps, F::NormalizeFuncOptions().p(2).dim(1));
    return F::linear(input, ww, bias);
}

AngularImpl::AngularImpl(int64_t in_features, int64_t out_features)
    : d_in(in_features)
    , d_out(out_features)
{
    reset();
}

void
AngularImpl::reset()
{
    weight = register_parameter("weight", torch::empty({ d_out, d_in }));
    bias = register_parameter("bias", {}, false);
    reset_parameters();
}

void
AngularImpl::reset_parameters()
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

TORCH_MODULE(Angular);
}
#endif
