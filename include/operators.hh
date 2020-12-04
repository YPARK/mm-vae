#ifndef TORCH_INC_H_                               // Prevent torch
#define TORCH_INC_H_                               // from reloading
#include <torch/torch.h>                           //
#include <torch/csrc/autograd/variable.h>          //
#include <torch/csrc/autograd/function.h>          //
#include <torch/csrc/autograd/VariableTypeUtils.h> //
#include <torch/csrc/autograd/functions/utils.h>   //
#endif                                             // End of torch

#ifndef OPERATORS_HH_
#define OPERATORS_HH_

struct LogModifiedBesselBackward : public torch::autograd::Node {

    using var_list = torch::autograd::variable_list;

    torch::autograd::SavedVariable self_; // To store the forward pass &
    at::Scalar df_;                       // To use it in gradient calculation

    var_list apply(var_list &&grads)
    {
        float df = df_.toDouble(); //
        auto &grad = grads[0];     // One output, so expect 1 gradient
        auto x = self_.unpack();   // Grab the data out of the saved variable
        var_list grad_inputs(1);   // Variable list to hold the gradients at
                                   // the function's input variables

        // Baricz et al. (2011) Lemma B.
        //
        // x I'(v,x)/I(v,x) > sqrt([v/(v +1)] * x^2  + v^2)
        //                  < sqrt(x^2 + v^2)
        if (should_compute_output(0)) {
            auto lb = torch::sqrt(x * x * df / (df + 1.) + df * df);
            auto ub = torch::sqrt(x * x + df * df);
            auto grad_result = 0.5 * (lb + ub) / x;
            grad_inputs[0] = grad_result; // only one output
        }

        return grad_inputs;
    }

    void release_variables()
    {
        self_.reset_data();          // destruction of saved variables
        self_.reset_grad_function(); // and gradients
    }
};

torch::Tensor
lbessel(const torch::Tensor self_, at::Scalar df_)
{
    using namespace torch::autograd;

    // Note: The method proposed by Gopal and Yang, ICML (2014) gives
    // a terrible bound for actual log Bessel evaluation
    //
    // But this works well: Oh, Adamczewski, Park (2019)
    //
    // (1) Provided eta = (df + 1/2) / (2 * (df + 1)), for kappa < df
    //
    // log(I(df, kappa)) = df * log(kappa) + eta * kappa - (eta + df) * log(2)
    //                 - log(Gamma(eta + 1))
    //
    // (2) For kappa > df,
    //
    // log(I(df, kappa)) = kappa - 0.5 * log(kappa) - 0.5 *log(2pi)
    //

    auto kappa = self_.data();
    auto nu = df_.toDouble();

    const float eta = (nu + 0.5) / (2. * (nu + 1.));

    auto stuff1 = nu * torch::log(kappa) + eta * kappa -
        (eta + nu) * std::log(2.) - fasterlgamma(nu + 1);

    auto stuff2 = kappa - 0.5 * torch::log(kappa) - 0.5 * std::log(2. * M_PI);

    auto result = torch::le(kappa, nu).type_as(kappa).mul(stuff1) +
        torch::gt(kappa, nu).type_as(kappa).mul(stuff2);

    // Prepare the infrastructure for computing the function's gradient
    if (compute_requires_grad(self_)) {
        // Initialize the gradient function
        auto grad_fn = std::shared_ptr<
            LogModifiedBesselBackward>(new LogModifiedBesselBackward(),
                                       deleteNode);

        // Connect into the autograd graph
        grad_fn->set_next_edges(collect_next_edges(self_));

        // Save the function arguments for use in the backwards pass
        grad_fn->self_ = SavedVariable(self_, false);
        grad_fn->df_ = df_;

        // Attach the gradient function to the result
        set_history(flatten_tensor_args(result), grad_fn);
    }

    return result;
}

#endif
