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
DIM(covar_dim);
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
        kappa_min = .1;
        kappa_max = 10.;
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
        "--kappa_min : max of the concentration parameter (default: .1)\n"
        "--kappa_max : min of the concentration parameter (default: 10)\n"
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
                           const covar_dim,
                           const z_repr_dim,
                           const std::vector<z_enc_dim>,
                           const std::vector<z_dec_dim>,
                           const kappa_min,
                           const kappa_max,
                           const bool);

    const int64_t x_dim;
    const int64_t c_dim;
    const int64_t z_dim;
    const bool do_relu;

    const float kap_min;
    const float kap_max;

    std::pair<torch::Tensor, torch::Tensor> encode(torch::Tensor x);

    std::pair<torch::Tensor, torch::Tensor> encode(torch::Tensor x,
                                                   torch::Tensor c);

    vmf_vae_out_t forward(torch::Tensor x, torch::Tensor c);

    torch::Tensor decode(torch::Tensor z, torch::Tensor c);

private:
    template <typename VEC>
    void _copy_dim_vec(const VEC &src, std::vector<int64_t> &dst);

    torch::Tensor reparameterize(torch::Tensor mu, torch::Tensor lnvar);

    torch::Tensor x_mean;
    torch::Tensor ln_x_sd;

    torch::Tensor ln_kappa;

    torch::nn::Linear covar_enc { nullptr };

    torch::nn::Sequential z_enc;
    torch::nn::Sequential z_dec;

    torch::nn::Linear covar_dec { nullptr };

    torch::nn::Linear z_repr_mean { nullptr };
    torch::nn::Linear z_repr_lnvar { nullptr };

    std::vector<int64_t> z_enc_dim_vec;
    std::vector<int64_t> z_dec_dim_vec;
};

std::pair<torch::Tensor, torch::Tensor>
vmf_vae_tImpl::encode(torch::Tensor x, torch::Tensor c)
{
    const float eps = 1e-2 / static_cast<float>(x.size(1));
    namespace F = torch::nn::functional;
    auto xn = F::normalize(x.log1p(), F::NormalizeFuncOptions().p(2).dim(1));

    auto xn_std =
        torch::div(torch::sub(xn, x_mean), F::softplus(ln_x_sd) + eps);

    auto h = z_enc->forward(xn_std); // x -> E[z]
    auto hc = covar_enc->forward(c); // c -> E[z]
    auto ln_var_clamp = torch::clamp(z_repr_lnvar->forward(h), -4., 4.);

    return { z_repr_mean->forward(h + hc), ln_var_clamp };
}

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
vmf_vae_tImpl::decode(torch::Tensor z, torch::Tensor c)
{
    namespace F = torch::nn::functional;
    auto h = torch::exp(z_dec->forward(z));
    auto hc = covar_dec->forward(c);
    return F::normalize(h + hc, F::NormalizeFuncOptions().p(2).dim(1));
}

vmf_vae_out_t
vmf_vae_tImpl::forward(torch::Tensor x, torch::Tensor c)
{
    auto enc_ = encode(x, c);
    auto mean_ = enc_.first;
    auto lnvar_ = enc_.second;

    auto recon = decode(reparameterize(mean_, lnvar_), c);

    auto kappa_clamp = torch::clamp(torch::exp(ln_kappa), kap_min, kap_max);

    return { recon, mean_, lnvar_, kappa_clamp };
}

/// Build the network
vmf_vae_tImpl::vmf_vae_tImpl(const data_dim xd_,
                             const covar_dim cd_,
                             const z_repr_dim z_,
                             const std::vector<z_enc_dim> ze_vec_,
                             const std::vector<z_dec_dim> zd_vec_,
                             const kappa_min _kmin = kappa_min { 1. },
                             const kappa_max _kmax = kappa_max { 100. },
                             const bool do_relu_ = false)
    : x_dim(xd_.val)
    , c_dim(cd_.val)
    , z_dim(z_.val)
    , kap_min(_kmin.val)
    , kap_max(_kmax.val)
    , do_relu(do_relu_)
    , x_mean(torch::zeros({ 1, x_dim }))
    , ln_x_sd(torch::ones({ 1, x_dim }))
    , ln_kappa(torch::ones({ 1 }) * std::log(kap_min))
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
        z_enc->push_back(mmvae::Angular(d_prev, d_next));
        if (do_relu)
            z_enc->push_back(torch::nn::ReLU(d_next));
        d_prev = d_next;
    }

    // Add final one mapping to the latent
    if (z_enc_dim_vec.size() < 1) {
        z_enc->push_back(mmvae::Angular(d_prev, z_dim));
        if (do_relu)
            z_enc->push_back(torch::nn::ReLU(z_dim));

        d_prev = z_dim;
    }

    covar_enc =
        register_module("covar: encoding", torch::nn::Linear(c_dim, z_dim));

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

    covar_dec =
        register_module("covar: decoding", torch::nn::Linear(c_dim, x_dim));
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
    auto yobs = F::normalize(F::relu(x).log1p() + eps,
                             F::NormalizeFuncOptions().p(2).dim(1));

    const float n = yobs.size(0);
    const float dd = yobs.size(1);
    const float df = std::max(0.5 * dd - 1., 0.);

    auto recon = yhat.recon; // N x D
    auto kappa = yhat.kappa; // scalar
    auto kl = kl_loss_normal(yhat.mean, yhat.lnvar);

    auto llik = torch::sum(yobs * recon, 1) * kappa;
    llik += df * torch::log(kappa) - lbessel(kappa, df);
    llik -= 0.5 * dd * fasterlog(2. * M_PI);

    return kl / n * kl_weight - llik.sum() / n;
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

struct vmf_vae_recorder_t {

    using Mat =
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    explicit vmf_vae_recorder_t(const std::string _hdr,
                                const int64_t _max_epoch)
        : header(_hdr)
        , epoch(0)
        , max_epoch(_max_epoch)
    {
    }

    template <typename MODEL, typename DB, typename BAT>
    void update_on_epoch(MODEL &model, DB &db, BAT &batches)
    {
        write(zeropad(++epoch, max_epoch));
    }

    template <typename MODEL, typename DB, typename BAT>
    void update_on_batch(MODEL &model, DB &db, BAT &batches)
    {
        model->train(false); // freeze the model

        const int64_t ntot = db.ntot();

        ///////////////////////
        // output mu encoder //
        ///////////////////////

        auto mu_out = model->encode(db.torch_tensor().to(torch::kCPU));
        torch::Tensor _mean = std::get<0>(mu_out);
        torch::Tensor _lnvar = std::get<1>(mu_out);

        if (mean_out.rows() < ntot || mean_out.cols() < _mean.size(1)) {
            mean_out.resize(ntot, _mean.size(1));
            mean_out.setZero();
        }

        if (lnvar_out.rows() < ntot || lnvar_out.cols() < _lnvar.size(1)) {
            lnvar_out.resize(ntot, _lnvar.size(1));
            lnvar_out.setZero();
        }

        Eigen::Map<Mat> _mean_mat(_mean.data_ptr<float>(),
                                  _mean.size(0),
                                  _mean.size(1));
        Eigen::Map<Mat> _lnvar_mat(_lnvar.data_ptr<float>(),
                                   _lnvar.size(0),
                                   _lnvar.size(1));

        for (int64_t j = 0; j < batches.size(); ++j) {
            if (batches[j] < ntot) {
                mean_out.row(batches[j]) = _mean_mat.row(j);
                lnvar_out.row(batches[j]) = _lnvar_mat.row(j);
            }
        }

        model->train(true); // release
    }

    void write(const std::string tag)
    {
        const std::string _hdr_tag =
            tag.size() > 0 ? (header + "_" + tag) : header;
        write_data_file(_hdr_tag + ".latent_mean.gz", mean_out);
        write_data_file(_hdr_tag + ".latent_lnvar.gz", lnvar_out);
    }

    const std::string header; // file header
    int64_t epoch;
    const int64_t max_epoch;
    Mat mean_out, lnvar_out;
};

TORCH_MODULE(vmf_vae_t); // expose vmf_vae_t ////////////////

}} // namespace
#endif
