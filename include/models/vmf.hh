#include "math.hh"
#include "operators.hh"
#include "angular.hh"

#ifndef MMVAE_VMF_MODEL_HH_
#define MMVAE_VMF_MODEL_HH_

namespace mmvae { namespace vmf {

const char *_model_desc = "Likelihood:\n"
                          "f(x) = C_{d}(κ) exp(κ μ'x)\n"
                          "where\n"
                          "\n              κ^{d/2 - 1}\n"
                          "C_{d}(κ) = -----------------------\n"
                          "           (2π)^{d/2} I_{d/2-1}(κ)\n"
                          "\n";

///////////////////////
// define dimensions //
///////////////////////

using pos_scalar_t = check_positive_t<float>;
using pos_size_t = check_positive_t<int64_t>;

#define SC(V)                     \
    struct V : pos_scalar_t {     \
        explicit V(const float x) \
            : pos_scalar_t(x)     \
        {                         \
        }                         \
    };

#define DIM(V)                  \
    struct V : pos_size_t {     \
        explicit V(const int x) \
            : pos_size_t(x)     \
        {                       \
        }                       \
    };

DIM(data_dim);
DIM(z_enc_dim);
DIM(z_repr_dim);
DIM(z_dec_dim);
DIM(kappa_net_dim);
SC(kappa_min);
SC(kappa_max);

//////////////////////////
// command line parsing //
//////////////////////////

struct vmf_options_t {

    explicit vmf_options_t()
    {
        // default options
        do_relu = false;
        latent = 2;
        kappa_min = .5;
        kappa_max = 500.;
    }

    std::vector<int64_t> encoding_layers;
    std::vector<int64_t> decoding_layers;
    int64_t latent;

    float kappa_min;
    float kappa_max;
    bool do_relu;
};

int
parse_vmf_options(const int argc, const char *_argv[], vmf_options_t &options)
{
    const char *_usage =
        "[von Mises Fisher VAE options]\n"
        "\n"
        "--encoding  : dims for encoding layers (e.g., 10,10)\n"
        "--decoding  : dims for decoding layers (e.g., 10,10)\n"
        "--latent    : latent z's dim\n"
        "--kappa_min : max of the concentration parameter (default: .5)\n"
        "--kappa_max : min of the concentration parameter (default: 500)\n"
        "--no_relu   : remove ReLU between layers (default)\n"
        "--relu      : add ReLU between layers\n"
        "\n";

    const char *const short_opts = "E:D:L:k:K:Rrh";

    const option long_opts[] = {
        { "encoding", required_argument, nullptr, 'E' },  //
        { "decoding", required_argument, nullptr, 'D' },  //
        { "latent", required_argument, nullptr, 'L' },    //
        { "kappa_min", required_argument, nullptr, 'k' }, //
        { "kappa-min", required_argument, nullptr, 'k' }, //
        { "kappa_max", required_argument, nullptr, 'K' }, //
        { "kappa-max", required_argument, nullptr, 'K' }, //
        { "relu", no_argument, nullptr, 'R' },            //
        { "no_relu", no_argument, nullptr, 'r' },         //
        { "no-relu", no_argument, nullptr, 'r' },         //
        { "help", no_argument, nullptr, 'h' },            //
        { nullptr, no_argument, nullptr, 0 }
    };

    auto copy_int_arr = [](const std::string src, std::vector<int64_t> &dst) {
        dst.clear();
        auto arr = split(src, ',');
        std::transform(arr.begin(),
                       arr.end(),
                       std::back_inserter(dst),
                       [](auto s) { return std::stol(s); });
    };

    optind = 1;
    opterr = 0;

    // copy argv and run over this instead of dealing with the actual ones.
    std::vector<const char *> _argv_org(_argv, _argv + argc);
    std::vector<const char *> argv_copy;
    std::transform(std::begin(_argv_org),
                   std::end(_argv_org),
                   std::back_inserter(argv_copy),
                   str2char);

    const char **argv = &argv_copy[0];

    while (true) {
        const auto opt = getopt_long(argc,                      //
                                     const_cast<char **>(argv), //
                                     short_opts,                //
                                     long_opts,                 //
                                     nullptr);

        if (-1 == opt)
            break;

        switch (opt) {
        case 'E':
            copy_int_arr(std::string(optarg), options.encoding_layers);
            break;

        case 'D':
            copy_int_arr(std::string(optarg), options.decoding_layers);
            break;

        case 'r':
            options.do_relu = false;
            break;

        case 'R':
            options.do_relu = true;
            break;

        case 'L':
            options.latent = std::stol(optarg);
            break;

        case 'k':
            options.kappa_min = std::stof(optarg);
            break;

        case 'K':
            options.kappa_max = std::stof(optarg);
            break;

        case 'h': // -h or --help
            std::cerr << _model_desc << std::endl;
            std::cerr << _usage << std::endl;
            for (std::size_t i = 0; i < argv_copy.size(); i++)
                delete[] argv_copy[i];
            return EXIT_SUCCESS;

        case '?': // Unrecognized option
            break;

        default: //
                 ;
        }
    }

    for (std::size_t i = 0; i < argv_copy.size(); i++)
        delete[] argv_copy[i];

    return EXIT_SUCCESS;
}

////////////////////////////////////////////
// Embedding von Mises Fisher to Gaussian //
////////////////////////////////////////////
struct vmf_vae_out_t {
    torch::Tensor recon;
    torch::Tensor mean;
    torch::Tensor lnvar;
    torch::Tensor kappa;
};

struct vmf_vae_tImpl : torch::nn::Module {

    explicit vmf_vae_tImpl(const data_dim,
                           const z_repr_dim,
                           const std::vector<z_enc_dim>,
                           const std::vector<z_dec_dim>,
                           const kappa_min,
                           const kappa_max,
                           const bool);

    const int64_t x_dim;
    const int64_t z_dim;
    const bool do_relu;

    const float kap_min;
    const float kap_max;

    std::pair<torch::Tensor, torch::Tensor> encode(torch::Tensor x);

    vmf_vae_out_t forward(torch::Tensor x);

    torch::Tensor decode(torch::Tensor z);

private:
    template <typename VEC>
    void _copy_dim_vec(const VEC &src, std::vector<int64_t> &dst);

    torch::Tensor reparameterize(torch::Tensor mu, torch::Tensor lnvar);

    torch::Tensor x_mean;
    torch::Tensor ln_x_sd;

    torch::Tensor ln_kappa;

    torch::nn::Sequential z_enc;
    torch::nn::Sequential z_dec;

    torch::nn::Linear z_repr_mean { nullptr };
    torch::nn::Linear z_repr_lnvar { nullptr };

    std::vector<int64_t> z_enc_dim_vec;
    std::vector<int64_t> z_dec_dim_vec;
};

std::pair<torch::Tensor, torch::Tensor>
vmf_vae_tImpl::encode(torch::Tensor x)
{
    const float eps = 1e-2 / static_cast<float>(x.size(1));
    namespace F = torch::nn::functional;
    auto xn = F::normalize(x.log1p(), F::NormalizeFuncOptions().p(2).dim(1));
    auto xn_std =
        torch::div(torch::sub(xn, x_mean), F::softplus(ln_x_sd) + eps);

    auto h = z_enc->forward(xn_std);

    auto ln_var_clamp = torch::clamp(z_repr_lnvar->forward(h), -4., 4.);

    return { z_repr_mean->forward(h), ln_var_clamp };
}

torch::Tensor
vmf_vae_tImpl::decode(torch::Tensor z)
{
    namespace F = torch::nn::functional;
    auto h = z_dec->forward(z);
    return F::normalize(h, F::NormalizeFuncOptions().p(2).dim(1));
}

vmf_vae_out_t
vmf_vae_tImpl::forward(torch::Tensor x)
{
    auto enc_ = encode(x);
    auto mean_ = enc_.first;
    auto lnvar_ = enc_.second;

    auto recon = decode(reparameterize(mean_, lnvar_));

    auto kappa_clamp = torch::clamp(torch::exp(ln_kappa), kap_min, kap_max);

    return { recon, mean_, lnvar_, kappa_clamp };
}

/// Build the network
vmf_vae_tImpl::vmf_vae_tImpl(const data_dim d_,
                             const z_repr_dim z_,
                             const std::vector<z_enc_dim> ze_vec_,
                             const std::vector<z_dec_dim> zd_vec_,
                             const kappa_min _kmin = kappa_min { 1. },
                             const kappa_max _kmax = kappa_max { 100. },
                             const bool do_relu_ = false)
    : x_dim(d_.val)
    , z_dim(z_.val)
    , kap_min(_kmin.val)
    , kap_max(_kmax.val)
    , do_relu(do_relu_)
    , x_mean(torch::zeros({ 1, x_dim }))
    , ln_x_sd(torch::ones({ 1, x_dim }))
    , ln_kappa(torch::ones({ 1 }) * (-z_dim))
{
    register_parameter("x_mean", x_mean);
    register_parameter("ln_x_sd", ln_x_sd);
    register_parameter("ln_kappa", ln_kappa);

    //////////////////////////////////
    // hidden encoding layers for z //
    //////////////////////////////////

    _copy_dim_vec(ze_vec_, z_enc_dim_vec);

    // Add hidden layers between x and representation
    int64_t d_prev = x_dim;

    for (int l = 0; l < z_enc_dim_vec.size(); ++l) {
        int64_t d_next = z_enc_dim_vec[l];
        z_enc->push_back(torch::nn::Linear(d_prev, d_next));
        if (do_relu)
            z_enc->push_back(torch::nn::ReLU(d_next));
        d_prev = d_next;
    }

    // Add final one mapping to the latent
    if (z_enc_dim_vec.size() < 1) {
        z_enc->push_back(torch::nn::Linear(d_prev, z_dim));
        if (do_relu)
            z_enc->push_back(torch::nn::ReLU(z_dim));

        d_prev = z_dim;
    }

    z_repr_mean = register_module("z: representation mean",
                                  torch::nn::Linear(d_prev, z_dim));

    z_repr_lnvar = register_module("z: representation log variance",
                                   torch::nn::Linear(d_prev, z_dim));

    //////////////////////////////////
    // hidden decoding layers for z //
    //////////////////////////////////

    _copy_dim_vec(zd_vec_, z_dec_dim_vec);

    // Add hidden layers between representation and x
    d_prev = z_dim;
    for (int l = 0; l < z_dec_dim_vec.size(); ++l) {
        int64_t d_next = z_dec_dim_vec[l];
        const std::string _str = "z: decoding " + std::to_string(l + 1);
        z_dec->push_back(torch::nn::Linear(d_prev, d_next));
        if (do_relu)
            z_dec->push_back(torch::nn::ReLU(d_next));
        d_prev = d_next;
    }

    // Add final one mapping to the reconstruction
    z_dec->push_back(torch::nn::Linear(d_prev, x_dim));
}

/// Gaussian reparameterization
/// @param mu
/// @param lnvar
torch::Tensor
vmf_vae_tImpl::reparameterize(torch::Tensor mu, torch::Tensor lnvar)
{
    if (is_training()) {
        auto sig = lnvar.div(2.0).exp();     //
        auto eps = torch::randn_like(lnvar); //
        return mu + eps.mul(sig);            //
    } else {
        return mu;
    }
}

/// KL-divergence loss from the standard Normal
/// Eq[ln q - ln p]
/// @param mean
/// @param lnvar
torch::Tensor
kl_loss_normal(torch::Tensor _mean, torch::Tensor _lnvar)
{
    return -0.5 * torch::sum(1 + _lnvar - _mean.pow(2) - _lnvar.exp());
}

/// @param x observed data
/// @param y model forward output
/// @param kl_weight up or down weight for KL divergence
torch::Tensor
vmf_vae_loss(torch::Tensor x, vmf_vae_out_t yhat, float kl_weight)
{
    const float eps = 1e-2 / static_cast<float>(x.size(1));
    namespace F = torch::nn::functional;
<<<<<<< HEAD
    auto yobs =
        F::normalize(x.log1p() + eps, F::NormalizeFuncOptions().p(2).dim(1));
=======
    auto yobs = F::normalize(x.log1p(), F::NormalizeFuncOptions().p(2).dim(1));
>>>>>>> 500fdf4d360fcd5afac8025c945fd8f3fd9d2d63

    const float n = yobs.size(0);
    const float dd = yobs.size(1);
    const float df = std::max(0.5 * dd - 1., 0.);

    auto recon = yhat.recon; // N x D
    auto kappa = yhat.kappa; // scalar
    auto kl = kl_loss_normal(yhat.mean, yhat.lnvar);

    auto llik = torch::sum(yobs * recon, 1) * kappa;
    llik += df * torch::log(kappa) - lbessel(kappa, df);
    // llik -= 0.5 * dd * fasterlog(2. * M_PI);

    return llik.sum() / n + kl / n * kl_weight;
}

//////////////////////
// helper functions //
//////////////////////

template <typename VEC>
void
vmf_vae_tImpl::_copy_dim_vec(const VEC &src, std::vector<int64_t> &dst)
{
    dst.clear();
    std::transform(src.begin(),
                   src.end(),
                   std::back_inserter(dst),
                   [](const auto x) { return x.val; });
}

TORCH_MODULE(vmf_vae_t); // expose vmf_vae_t ////////////////

//////////////////////////////
// von Mises Fisher mixture //
//////////////////////////////

struct vmf_mixture_tImpl : torch::nn::Module {

    vmf_mixture_tImpl(torch::Tensor _label,
                      const kappa_min _kmin,
                      const kappa_max _kmax);

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

    const torch::Tensor L; // D x K

    const int64_t D;
    const int64_t K;

    torch::Tensor take_estep(torch::Tensor x);

    float dd;

    torch::Tensor ln_mu;       // D x K parameter
    torch::Tensor mu;          // D x K parameter
    torch::Tensor logit_kappa; // 1 x 1 parmeter
    torch::Tensor kappa;       // 1 x 1 parmeter
    torch::Tensor filter;      // 1 x D

    const float kap_min;
    const float kap_max;

    void resolve_mu();
    void resolve_kappa();

    torch::Tensor normalize_x(torch::Tensor x);
};

vmf_mixture_tImpl::vmf_mixture_tImpl(torch::Tensor _label,
                                     const kappa_min _kmin = kappa_min { 1. },
                                     const kappa_max _kmax = kappa_max { 100. })
    : L(_label)
    , D(L.size(0))
    , K(L.size(1))
    , ln_mu(torch::zeros({ D, K }))
    , mu(torch::zeros({ D, K }))
    , logit_kappa(torch::ones({ 1 }) * (-fasterlog(K)))
    , kappa(torch::zeros({ 1 }))
    , filter(torch::zeros({ 1, D }))
    , kap_min(_kmin.val)
    , kap_max(_kmax.val)
{
    register_parameter("ln_mu", ln_mu);
    register_parameter("logit_kappa", logit_kappa);
    register_parameter("mu", mu);
    register_parameter("kappa", kappa);

    // Ask how many features are effectively non-zero?
    filter =
        torch::mm(torch::ones({ 1, K }), L.transpose(0, 1)).gt(0.).type_as(L);

    dd = filter.sum().item<float>();

    TLOG("Dimensionality in vMF mixture: " << dd);

    resolve_mu();
    resolve_kappa();
}

std::tuple<torch::Tensor, torch::Tensor>
vmf_mixture_tImpl::forward(torch::Tensor x)
{
    namespace F = torch::nn::functional;

    auto z = take_estep(x).detach();
    auto xn = normalize_x(x);

    const float df = std::max(0.5 * dd - 1., 0.);

    auto ln_q_ = torch::mm(xn, mu) * kappa;
    auto llik = ln_q_ + df * torch::log(kappa) - lbessel(kappa, df);
    llik -= 0.5 * dd * fasterlog(2. * M_PI);

    return std::make_tuple(torch::sum(-llik * z, 1), ln_q_);
}

//////////////////////////////////////////////////////////
// Make sure that the data can reside on angular domain //
//////////////////////////////////////////////////////////

/// @param x
torch::Tensor
vmf_mixture_tImpl::normalize_x(torch::Tensor x)
{
    const float eps = 1e-4;
    namespace F = torch::nn::functional;
    auto opt_ = F::NormalizeFuncOptions().p(2).dim(1);

    return F::normalize((x.log1p() + eps).mul(filter), opt_);
}

void
vmf_mixture_tImpl::resolve_mu()
{
    const float eps = 1e-4;
    namespace F = torch::nn::functional;
    mu = F::normalize(ln_mu.exp().mul(L) + eps,
                      F::NormalizeFuncOptions().p(2).dim(0));
}

void
vmf_mixture_tImpl::resolve_kappa()
{
    const float delt = kap_max - kap_min;
    kappa = delt * torch::sigmoid(logit_kappa) + kap_min;
}

torch::Tensor
vmf_mixture_tImpl::take_estep(torch::Tensor x)
{
    resolve_mu();
    resolve_kappa();
    namespace F = torch::nn::functional;
    auto xn = normalize_x(x);
    auto logits = torch::log_softmax(torch::mm(xn, mu) * kappa, 1);

    if (!is_training()) {
        return logits.exp();
    }

    return F::gumbel_softmax(logits,
                             F::GumbelSoftmaxFuncOptions().dim(1).hard(true));
}

/// KL-divergence loss from the uniform prior
/// Eq[ln q - ln p]
/// @param ln_q_y variational log probability
torch::Tensor
kl_loss_uniform(torch::Tensor ln_q_)
{
    auto ln_z = torch::log_softmax(ln_q_, 1);
    const float K = ln_z.size(1);
    return torch::sum(ln_z.exp() * (ln_z + fasterlog(K)), 1);
}

TORCH_MODULE(vmf_mixture_t); // expose vmf_mixture_t

}} // namespace
#endif
