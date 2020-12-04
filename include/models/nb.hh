#ifndef TORCH_INC_H_     // Prevent torch
#define TORCH_INC_H_     // from reloading
#include <torch/torch.h> //
#endif                   // End of torch

#include "util.hh"
#include "std_util.hh"
#include <getopt.h>
#include <cstring>
#include "angular.hh"

#ifndef MMVAE_NB_MODEL_HH_
#define MMVAE_NB_MODEL_HH_

namespace mmvae { namespace nb {

const char *_model_desc = "Likelihood:\n"
                          "     Γ(x + ν)        μ           ν   \n"
                          "x ~ ------------ ( ----- )^x ( ----- )^ν\n"
                          "    Γ(x + 1)Γ(ν)   μ + ν       μ + ν\n"
                          "\n"
                          "μ = exp(decoding(z_μ) + bias_μ)\n"
                          "ν = exp(decoding(z_ν) + bias_ν)\n"
                          "\n";

///////////////////////
// define dimensions //
///////////////////////

using pos_size_t = check_positive_t<int64_t>;

#define DIM(V)                  \
    struct V : pos_size_t {     \
        explicit V(const int x) \
            : pos_size_t(x)     \
        {                       \
        }                       \
    };

DIM(data_dim);
DIM(mu_encoder_h_dim);
DIM(mu_decoder_h_dim);
DIM(mu_encoder_r_dim);
DIM(nu_encoder_h_dim);
DIM(nu_encoder_r_dim);

//////////////////////////
// command line parsing //
//////////////////////////

struct nbvae_options_t {

    explicit nbvae_options_t()
    {
        // default options
        do_relu = false;
        mean_latent = 2;
        overdispersion_encoding = 1;
        overdispersion_latent = 1;
    }

    std::vector<int64_t> mean_encoding_layers;
    std::vector<int64_t> mean_decoding_layers;
    int64_t mean_latent;
    int64_t overdispersion_encoding;
    int64_t overdispersion_latent;
    bool do_relu;
};

int
parse_nbvae_options(const int argc,
                    const char *_argv[],
                    nbvae_options_t &options)
{
    const char *_usage =
        "[Negative Binomial VAE options]\n"
        "\n"
        "--mean_encoding     : dims for mean encoding layers (e.g., 10,10)\n"
        "--mean_decoding     : dims for mean decoding layers (e.g., 10,10)\n"
        "                    : If the final dim does not match with data\n"
        "                    : it will create an additional layer.\n"
        "\n"
        "--mean_latent       : latent z's dim for the mean\n"
        "\n"
        "--overdisp_encoding : dim for overdispersion encoding (default: 1)\n"
        "\n"
        "--overdisp_latent   : latent dim for the overdispersion (default: 1)\n"
        "--no_relu           : remove ReLU between layers (default)\n"
        "--relu              : add ReLU between layers\n"
        "\n";

    const char *const short_opts = "E:D:L:e:l:rRh";

    const option long_opts[] = {
        { "mean_encoding", required_argument, nullptr, 'E' },           //
        { "mean-encoding", required_argument, nullptr, 'E' },           //
        { "mean_decoding", required_argument, nullptr, 'D' },           //
        { "mean-decoding", required_argument, nullptr, 'D' },           //
        { "mean_latent", required_argument, nullptr, 'L' },             //
        { "mean-latent", required_argument, nullptr, 'L' },             //
        { "overdisp_encoding", required_argument, nullptr, 'e' },       //
        { "overdisp-encoding", required_argument, nullptr, 'e' },       //
        { "overdisp_decoding", required_argument, nullptr, 'e' },       //
        { "overdisp-decoding", required_argument, nullptr, 'e' },       //
        { "overdispersion_encoding", required_argument, nullptr, 'e' }, //
        { "overdispersion-encoding", required_argument, nullptr, 'e' }, //
        { "overdispersion_decoding", required_argument, nullptr, 'e' }, //
        { "overdispersion-decoding", required_argument, nullptr, 'e' }, //
        { "relu", no_argument, nullptr, 'R' },                          //
        { "no_relu", no_argument, nullptr, 'r' },                       //
        { "no-relu", no_argument, nullptr, 'r' },                       //
        { "help", no_argument, nullptr, 'h' },                          //
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
            copy_int_arr(std::string(optarg), options.mean_encoding_layers);
            break;

        case 'D':
            copy_int_arr(std::string(optarg), options.mean_decoding_layers);
            break;

        case 'L':
            options.mean_latent = std::stol(optarg);
            break;

        case 'e':
            options.overdispersion_encoding = std::stol(optarg);
            break;

        case 'l':
            options.overdispersion_latent = std::stol(optarg);
            break;

        case 'r':
            options.do_relu = false;
            break;

        case 'R':
            options.do_relu = true;
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

////////////////////////////////////////////////
// Negative Binomial Variational Auto Encoder //
////////////////////////////////////////////////

struct nbvae_out_t {
    using T = torch::Tensor;

    T recon_mu;
    T recon_nu;
    T mu_mean;
    T mu_lnvar;
    T nu_mean;
    T nu_lnvar;
};

struct nbvae_tImpl : torch::nn::Module {

    explicit nbvae_tImpl(const data_dim,
                         const std::vector<mu_encoder_h_dim>,
                         const std::vector<mu_decoder_h_dim>,
                         const mu_encoder_r_dim,
                         const nu_encoder_h_dim,
                         const nu_encoder_r_dim,
                         const bool _do_relu);

    // Encoding the mu of NB model -> mean, lnvar (mean)
    std::pair<torch::Tensor, torch::Tensor> encode_mu(torch::Tensor x);

    // Encoding the nu of NB model -> mean, lnvar (overdispersion)
    std::pair<torch::Tensor, torch::Tensor> encode_nu(torch::Tensor x);

    // Decoding the mu of NB model
    torch::Tensor decode_mu(torch::Tensor z);

    // Decoding the nu of NB model
    torch::Tensor decode_nu(torch::Tensor z);

    // Forward pass: mu (log), nu (log)
    nbvae_out_t forward(torch::Tensor x);

    const int64_t x_dim;
    const int64_t mu_r_dim;
    const int64_t nu_h_dim;
    const int64_t nu_r_dim;

    int64_t dim_data() const { return x_dim; }
    int64_t dim_mu_latent() const { return mu_r_dim; }
    int64_t dim_nu_latent() const { return nu_r_dim; }

    const float nu_max = 1e4;

    const bool do_relu;

private:
    template <typename VEC>
    void _copy_dim_vec(const VEC &src, std::vector<int64_t> &dst);

    torch::Tensor reparameterize(torch::Tensor mu, torch::Tensor lnvar);

    torch::Tensor x_mean;
    torch::Tensor ln_x_sd;

    torch::Tensor mu_bias; // 1 x D
    torch::Tensor nu_bias; // 1 x D

    torch::nn::Sequential mu_enc;

    torch::nn::Linear mu_repr_mean { nullptr };
    torch::nn::Linear mu_repr_lnvar { nullptr };

    torch::nn::Sequential mu_dec;

    torch::nn::Linear nu_enc { nullptr };

    torch::nn::Linear nu_repr_mean { nullptr };
    torch::nn::Linear nu_repr_lnvar { nullptr };
    torch::nn::Linear nu_dec { nullptr };

    std::vector<int64_t> mu_eh_dim_vec;
    std::vector<int64_t> mu_dh_dim_vec;
};

/// loss function for NB-VAE
/// @param x observed data
/// @param y model forward output
/// @param kl_weight up or down weight for KL divergence
torch::Tensor loss(torch::Tensor x, nbvae_out_t y, float kl_weight);

/////////////
// details //
/////////////

nbvae_tImpl::nbvae_tImpl(const data_dim _d,
                         const std::vector<mu_encoder_h_dim> _eh_vec,
                         const std::vector<mu_decoder_h_dim> _dh_vec,
                         const mu_encoder_r_dim _mr,
                         const nu_encoder_h_dim _oh,
                         const nu_encoder_r_dim _or,
                         const bool _do_relu = true)
    : x_dim(_d.val)
    , mu_r_dim(_mr.val)
    , nu_h_dim(_oh.val)
    , nu_r_dim(_or.val)
    , x_mean(torch::zeros({ 1, x_dim }))
    , ln_x_sd(torch::ones({ 1, x_dim }))
    , mu_bias(torch::zeros({ 1, x_dim }))
    , nu_bias(torch::zeros({ 1, x_dim }))
    , do_relu(_do_relu)
{
    register_parameter("x_mean", x_mean);
    register_parameter("ln_x_sd", ln_x_sd);

    register_parameter("mu_bias", mu_bias);
    register_parameter("nu_bias", nu_bias);

    ///////////////////////////////////
    // hidden encoding layers for mu //
    ///////////////////////////////////

    // ASSERT(_eh_vec.size() > 0, "at least one dim info for a hidden
    // layer(s)");
    _copy_dim_vec(_eh_vec, mu_eh_dim_vec);

    // Add hidden layers between x and representation
    int64_t d_prev = x_dim;
    for (int l = 0; l < mu_eh_dim_vec.size(); ++l) {
        int64_t d_next = mu_eh_dim_vec[l];
        mu_enc->push_back(torch::nn::Linear(d_prev, d_next));
        if (do_relu)
            mu_enc->push_back(torch::nn::ReLU(d_next));
        d_prev = d_next;
    }

    // Add final one mapping to the latent
    if (mu_eh_dim_vec.size() < 1) {
        mu_enc->push_back(torch::nn::Linear(d_prev, mu_r_dim));
        if (do_relu)
            mu_enc->push_back(torch::nn::ReLU(mu_r_dim));

        d_prev = mu_r_dim;
    }

    mu_repr_mean = register_module("mu: representation mean",
                                   torch::nn::Linear(d_prev, mu_r_dim));

    mu_repr_lnvar = register_module("mu: representation log variance",
                                    torch::nn::Linear(d_prev, mu_r_dim));

    ///////////////////////////////////
    // hidden decoding layers for mu //
    ///////////////////////////////////

    // ASSERT(_dh_vec.size() > 0, "at least one dim info for a hidden
    // layer(s)");
    _copy_dim_vec(_dh_vec, mu_dh_dim_vec);

    // Add hidden layers between representation and x
    d_prev = mu_r_dim;
    for (int l = 0; l < mu_dh_dim_vec.size(); ++l) {
        int64_t d_next = mu_dh_dim_vec[l];
        const std::string _str = "mu: decoding " + std::to_string(l + 1);
        mu_dec->push_back(torch::nn::Linear(d_prev, d_next));
        if (do_relu)
            mu_dec->push_back(torch::nn::ReLU(d_next));
        d_prev = d_next;
    }

    // Add final one mapping to the reconstruction
    mu_dec->push_back(torch::nn::Linear(d_prev, x_dim));

    ///////////////////////////////////
    // hidden encoding layers for nu //
    ///////////////////////////////////

    nu_enc =
        register_module("nu: encoding", torch::nn::Linear(x_dim, nu_h_dim));

    nu_repr_mean = register_module("nu: representation mean",
                                   torch::nn::Linear(nu_h_dim, nu_r_dim));
    nu_repr_lnvar = register_module("nu: representation log variance",
                                    torch::nn::Linear(nu_h_dim, nu_r_dim));
    nu_dec =
        register_module("nu: decoding", torch::nn::Linear(nu_r_dim, x_dim));
}

std::pair<torch::Tensor, torch::Tensor>
nbvae_tImpl::encode_mu(torch::Tensor x)
{
    const float eps = 1e-2;

    namespace F = torch::nn::functional;
    auto xn = F::normalize(x.log1p(), F::NormalizeFuncOptions().p(2).dim(1));
    auto xn_std =
        torch::div(torch::sub(xn, x_mean), F::softplus(ln_x_sd) + eps);

    auto h = mu_enc->forward(xn_std);
    auto ln_var_clamp = torch::clamp(mu_repr_lnvar->forward(h), -4., 4.);

    return { mu_repr_mean->forward(h), ln_var_clamp };
}

torch::Tensor
nbvae_tImpl::decode_mu(torch::Tensor z)
{
    auto h = mu_dec->forward(z);
    return torch::exp(h + mu_bias);
}

std::pair<torch::Tensor, torch::Tensor>
nbvae_tImpl::encode_nu(torch::Tensor x)
{
    namespace F = torch::nn::functional;
    auto h = F::relu(nu_enc->forward(x));

    auto ln_var_clamp = torch::clamp(nu_repr_lnvar->forward(h), -4., 4.);

    return { nu_repr_mean->forward(h), ln_var_clamp };
}

torch::Tensor
nbvae_tImpl::decode_nu(torch::Tensor z)
{
    auto ret = torch::exp(nu_dec->forward(z) - nu_bias);
    return torch::clamp(ret, 0., nu_max);
}

torch::Tensor
nbvae_tImpl::reparameterize(torch::Tensor mu, torch::Tensor lnvar)
{
    if (is_training()) {
        auto sig = lnvar.div(2.0).exp();     //
        auto eps = torch::randn_like(lnvar); //
        return mu + eps.mul(sig);            //
    } else {
        return mu;
    }
}

nbvae_out_t
nbvae_tImpl::forward(torch::Tensor x)
{
    ///////////////////////////
    // Reparameterization mu //
    ///////////////////////////
    auto mu_enc_out = encode_mu(x);
    auto mu_mean = mu_enc_out.first;
    auto mu_lnvar = mu_enc_out.second;
    auto _mu = decode_mu(reparameterize(mu_mean, mu_lnvar));

    /////////////////////
    // over-dispersion //
    /////////////////////

    auto nu_enc_out = encode_nu(x);
    auto nu_mean = nu_enc_out.first;
    auto nu_lnvar = nu_enc_out.second;
    auto _nu = decode_nu(reparameterize(nu_mean, nu_lnvar));

    return { _mu, _nu, mu_mean, mu_lnvar, nu_mean, nu_lnvar };
}

/// Negative Binomial Loss
torch::Tensor
nllik_loss(torch::Tensor x, nbvae_out_t y)
{
    const float eps = 1e-4;

    namespace F = torch::nn::functional;

    auto xn = F::normalize(x, F::NormalizeFuncOptions().p(2).dim(1)).mul(1e4);

    auto nu = y.recon_nu + eps;
    auto mu = y.recon_mu + eps;

    // log-gamma ratio
    auto lg = torch::lgamma(nu) + torch::lgamma(xn + 1.);
    lg -= torch::lgamma(nu + xn);

    // logit-like llik
    auto denom = torch::log(mu + nu);
    auto pr = xn.mul(denom - torch::log(mu));
    pr += nu.mul(denom - torch::log(nu));

    return torch::sum(lg + pr);
}

torch::Tensor
kl_loss(torch::Tensor _mean, torch::Tensor _lnvar)
{
    return -0.5 * torch::sum(1 + _lnvar - _mean.pow(2) - _lnvar.exp());
}

torch::Tensor
loss(torch::Tensor x, nbvae_out_t y, float kl_weight = 1.)
{
    auto recon_loss = nllik_loss(x, y);
    const float n = x.size(0);
    auto ret = recon_loss / n;
    ret += kl_loss(y.mu_mean, y.mu_lnvar) * kl_weight / n;
    ret += kl_loss(y.nu_mean, y.nu_lnvar) * kl_weight / n;
    return ret;
}

//////////////////////
// helper functions //
//////////////////////

template <typename VEC>
void
nbvae_tImpl::_copy_dim_vec(const VEC &src, std::vector<int64_t> &dst)
{
    dst.clear();
    std::transform(src.begin(),
                   src.end(),
                   std::back_inserter(dst),
                   [](const auto x) { return x.val; });
}

TORCH_MODULE(nbvae_t); // expose nbvae_t
}}                     // namespace
#endif
