#include "math.hh"
#include "operators.hh"
#include "angular.hh"

#ifndef MMVAE_VMFNB_MODEL_HH_
#define MMVAE_VMFNB_MODEL_HH_

namespace mmvae { namespace vmfnb {

const char *_model_desc = "[Likelihood]\n"
                          "\n"
                          "We model both f(x) and f(y) jointly...\n"
                          "\n"
                          "        Γ(x + ν)        μ            ν   \n"
                          "f(x) = ------------ ( ------ )^x ( ------ )^ν\n"
                          "       Γ(x + 1)Γ(ν)   μ + ν        μ + ν\n"
                          "\n"
                          "μ = exp(decoding(z_μ) + bias_μ)\n"
                          "ν = exp(decoding(z_ν) + bias_ν)\n"
                          "\n"
                          "f(y) = C_{d}(κ) exp(κ θ'y)\n"
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
DIM(mu_nb_enc_h_dim);
DIM(mu_nb_dec_h_dim);
DIM(mu_nb_enc_r_dim);
DIM(nu_nb_enc_h_dim);
DIM(nu_nb_dec_h_dim);
DIM(nu_nb_enc_r_dim);
SC(kappa_min);
SC(kappa_max);

//////////////////////////
// command line parsing //
//////////////////////////

struct vmfnb_options_t {

    explicit vmfnb_options_t()
    {
        // default options
        do_relu = false;
        mean_latent = 2;
        overdispersion_encoding = 1;
        overdispersion_latent = 1;
        kappa_min = .1;
        kappa_max = 100.;
    }

    std::vector<int64_t> mean_encoding_layers;
    std::vector<int64_t> mean_decoding_layers;
    int64_t mean_latent;

    int64_t overdispersion_encoding;
    int64_t overdispersion_latent;

    float kappa_min;
    float kappa_max;
    bool do_relu;
};

int
parse_vmfnb_options(const int argc,
                    const char *_argv[],
                    vmfnb_options_t &options)
{
    const char *_usage =
        "[von Mises Fisher + Negative Binomial VAE options]\n"
        "\n"
        "--mean_encoding     : dims for mean encoding layers (e.g., 10,10)\n"
        "--mean_decoding     : dims for mean decoding layers (e.g., 10,10)\n"
        "--mean_latent       : latent z's dim for the mean\n"
        "\n"
        "--overdisp_encoding : dim for overdispersion encoding (e.g., 3,3)\n"
        "--overdisp_latent   : latent dim for the overdispersion (default: 1)\n"
        "\n"
        "--kappa_min         : max of the concentration parameter (default: .1)\n"
        "--kappa_max         : min of the concentration parameter (default: 100)\n"
        "--no_relu           : remove ReLU between layers (default)\n"
        "--relu              : add ReLU between layers\n"
        "\n";

    const char *const short_opts = "E:D:L:k:K:Rrh";

    const option long_opts[] = {
        { "mean_encoding", required_argument, nullptr, 'E' },           //
        { "mean-encoding", required_argument, nullptr, 'E' },           //
        { "mean_decoding", required_argument, nullptr, 'D' },           //
        { "mean-decoding", required_argument, nullptr, 'D' },           //
        { "mean_latent", required_argument, nullptr, 'L' },             //
        { "mean-latent", required_argument, nullptr, 'L' },             //
        { "overdisp_encoding", required_argument, nullptr, 'e' },       //
        { "overdisp-encoding", required_argument, nullptr, 'e' },       //
        { "overdispersion_encoding", required_argument, nullptr, 'e' }, //
        { "overdispersion-encoding", required_argument, nullptr, 'e' }, //
        { "overdispersion_latent", required_argument, nullptr, 'l' },   //
        { "overdispersion-latent", required_argument, nullptr, 'l' },   //
        { "kappa_min", required_argument, nullptr, 'k' },               //
        { "kappa-min", required_argument, nullptr, 'k' },               //
        { "kappa_max", required_argument, nullptr, 'K' },               //
        { "kappa-max", required_argument, nullptr, 'K' },               //
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

        case 'k':
            options.kappa_min = std::stof(optarg);
            break;

        case 'K':
            options.kappa_max = std::stof(optarg);
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

///////////////////////////////
// von Mises Fisher + DB VAE //
///////////////////////////////

struct vmfnb_vae_out_t {
    using T = torch::Tensor;

    T nb_recon_mu;
    T nb_recon_nu;
    T nb_recon_depth;

    T nb_mu_mean;  // aggregate mean for mean
    T nb_mu_lnvar; // shared log variance for mean
    T nb_nu_mean;  // overdisp mean
    T nb_nu_lnvar; // overdisp log variance

    T vmf_recon;
    T vmf_logits;
    T vmf_kappa;

    float dd;
};

struct vmf_estep_out_t {
    using T = torch::Tensor;
    T logits;
    T latent;
};

struct vmf_out_t {
    using T = torch::Tensor;

    T mu;
    T logits;
    T latent;
    T recon;
    T kappa;

    float dd;
};

struct vmfnb_vae_tImpl : torch::nn::Module {

    vmfnb_vae_tImpl(torch::Tensor _label,
                    const std::vector<mu_nb_enc_h_dim>,
                    const std::vector<mu_nb_dec_h_dim>,
                    const mu_nb_enc_r_dim,
                    const nu_nb_enc_h_dim,
                    const nu_nb_enc_r_dim,
                    const kappa_min _kmin,
                    const kappa_max _kmax,
                    const bool _do_relu);

    vmfnb_vae_out_t forward(torch::Tensor x);

    std::pair<torch::Tensor, torch::Tensor> nb_encode_mu(torch::Tensor x,
                                                         torch::Tensor z);
    torch::Tensor nb_decode_mu(torch::Tensor z);

    std::pair<torch::Tensor, torch::Tensor> nb_encode_nu(torch::Tensor x);
    torch::Tensor nb_decode_nu(torch::Tensor z);

    torch::Tensor reparameterize(torch::Tensor mu, torch::Tensor lnvar);

    torch::Tensor nb_encode_mu_k(torch::Tensor x, const int64_t k);

    const torch::Tensor L; // D x K
    const int64_t D;
    const int64_t K;

    //////////////////////
    // NB-related stuff //
    //////////////////////

    torch::Tensor x_mean;  // 1 x D
    torch::Tensor ln_x_sd; // 1 x D

    torch::Tensor mu_bias; // 1 x D
    torch::Tensor nu_bias; // 1 x D

    torch::nn::Sequential nb_mu_enc;

    // K latent representations, one for each k
    torch::nn::ModuleList nb_mu_repr_mean_list;
    // Shared variance parameter
    torch::nn::Linear nb_mu_repr_lnvar { nullptr };
    torch::Tensor nb_mu_repr_mean;

    torch::nn::Sequential nb_mu_dec;

    // Overdispersion for NB
    torch::nn::Linear nb_nu_enc { nullptr };

    torch::nn::Linear nb_nu_repr_mean { nullptr };
    torch::nn::Linear nb_nu_repr_lnvar { nullptr };
    torch::nn::Linear nb_nu_dec { nullptr };

    torch::nn::Linear depth { nullptr };

    ///////////////////////
    // vMF-related stuff //
    ///////////////////////

    vmf_out_t vmf_forward(torch::Tensor x);

    vmf_estep_out_t
    take_vmf_estep(torch::Tensor x, torch::Tensor mu, torch::Tensor kappa);

    float dd;

    torch::Tensor ln_vmf_mu;                // D x K parameter
    torch::nn::Linear ln_kappa { nullptr }; // n x 1
    torch::Tensor filter;                   // 1 x D

    const float kap_min;
    const float kap_max;
    const bool do_relu;

    const float nu_max = 1e4;

    torch::Tensor normalize_vmf_x(torch::Tensor x);
    torch::Tensor normalize_nb_x(torch::Tensor x);
};

/////////////
// details //
/////////////

vmfnb_vae_tImpl::vmfnb_vae_tImpl(torch::Tensor _label,
                                 const std::vector<mu_nb_enc_h_dim> mu_nb_enc,
                                 const std::vector<mu_nb_dec_h_dim> mu_nb_dec,
                                 const mu_nb_enc_r_dim mu_nb_repr,
                                 const nu_nb_enc_h_dim nu_nb_enc,
                                 const nu_nb_enc_r_dim nu_nb_repr,
                                 const kappa_min _kmin = kappa_min { 1. },
                                 const kappa_max _kmax = kappa_max { 100. },
                                 const bool _do_relu = false)
    : L(_label)
    , D(L.size(0))
    , K(L.size(1))
    , x_mean(torch::zeros({ 1, D }))
    , ln_x_sd(torch::ones({ 1, D }))
    , mu_bias(torch::zeros({ 1, D }))
    , nu_bias(torch::zeros({ 1, D }))
    , ln_vmf_mu(torch::zeros({ D, K }))
    , filter(torch::zeros({ 1, D }))
    , kap_min(_kmin.val)
    , kap_max(_kmax.val)
    , do_relu(_do_relu)
{

    register_parameter("x_mean", x_mean);
    register_parameter("ln_x_sd", ln_x_sd);

    register_parameter("mu_bias", mu_bias);
    register_parameter("nu_bias", nu_bias);

    ////////////////////////////////////////
    // initialize the network for NB path //
    ////////////////////////////////////////

    // Add hidden layers between x and representation
    int64_t d_prev = D;
    for (int l = 0; l < mu_nb_enc.size(); ++l) {
        int64_t d_next = mu_nb_enc[l].val;
        nb_mu_enc->push_back(torch::nn::Linear(d_prev, d_next));
        if (do_relu)
            nb_mu_enc->push_back(torch::nn::ReLU(d_next));
        d_prev = d_next;
    }

    if (mu_nb_enc.size() < 1) {
        nb_mu_enc->push_back(torch::nn::Linear(d_prev, mu_nb_repr.val));
        if (do_relu)
            nb_mu_enc->push_back(torch::nn::ReLU(mu_nb_repr.val));

        d_prev = mu_nb_repr.val;
    }

    // K components for Gaussian representation
    for (int k = 0; k < K; ++k) {
        nb_mu_repr_mean_list->push_back(
            torch::nn::Linear(d_prev, mu_nb_repr.val));
    }

    // shared variance across all the components
    nb_mu_repr_lnvar =
        register_module("[NB] mu: representation log variance",
                        torch::nn::Linear(d_prev, mu_nb_repr.val));

    // Add hidden layers between representation and x
    d_prev = mu_nb_repr.val;
    for (int l = 0; l < mu_nb_dec.size(); ++l) {
        int64_t d_next = mu_nb_dec[l].val;
        const std::string _str = "mu: decoding " + std::to_string(l + 1);
        nb_mu_dec->push_back(torch::nn::Linear(d_prev, d_next));
        if (do_relu)
            nb_mu_dec->push_back(torch::nn::ReLU(d_next));
        d_prev = d_next;
    }

    // Add final one mapping to the reconstruction
    nb_mu_dec->push_back(torch::nn::Linear(d_prev, D));

    ///////////////////////////////////
    // hidden encoding layers for nu //
    ///////////////////////////////////

    nb_nu_enc = register_module("[NB] nu: encoding",
                                torch::nn::Linear(D, nu_nb_enc.val));

    nb_nu_repr_mean =
        register_module("[NB] nu: representation mean",
                        torch::nn::Linear(nu_nb_enc.val, nu_nb_repr.val));
    nb_nu_repr_lnvar =
        register_module("[NB] nu: representation log variance",
                        torch::nn::Linear(nu_nb_enc.val, nu_nb_repr.val));
    nb_nu_dec = register_module("[NB] nu: decoding",
                                torch::nn::Linear(nu_nb_repr.val, D));

    //////////////////////
    // sequencing depth //
    //////////////////////

    depth = register_module("[NB] depth", torch::nn::Linear(D, 1));

    /////////////////////////
    // initialize vMF path //
    /////////////////////////

    register_parameter("[vMF] ln mu", ln_vmf_mu);
    ln_kappa = register_module("[vMF] kappa", torch::nn::Linear(D, 1));

    // Ask how many features are effectively non-zero?
    filter =
        torch::mm(torch::ones({ 1, K }), L.transpose(0, 1)).gt(0.).type_as(L);

    dd = filter.sum().item<float>();

    TLOG("Dimensionality in vMF mixture: " << dd);
}

torch::Tensor
vmfnb_vae_tImpl::nb_encode_mu_k(torch::Tensor x, const int64_t k)
{
    namespace F = torch::nn::functional;

    auto xn_std = normalize_nb_x(x);
    auto h = nb_mu_enc->forward(xn_std);

    auto mu_k_ = nb_mu_repr_mean_list->ptr<torch::nn::LinearImpl>(k);

    return mu_k_->forward(h);
}

std::pair<torch::Tensor, torch::Tensor>
vmfnb_vae_tImpl::nb_encode_mu(torch::Tensor x, torch::Tensor z)
{
    namespace F = torch::nn::functional;

    auto xn_std = normalize_nb_x(x);
    auto h = nb_mu_enc->forward(xn_std);
    auto ln_var_clamp = torch::clamp(nb_mu_repr_lnvar->forward(h), -4., 4.);

    auto mu_0 = nb_mu_repr_mean_list->ptr<torch::nn::LinearImpl>(0);
    auto mu = mu_0->forward(h) * z.slice(1, 0, 1);

    for (int k = 1; k < K; ++k) {
        auto mu_k = nb_mu_repr_mean_list->ptr<torch::nn::LinearImpl>(k);
        mu += mu_k->forward(h) * z.slice(1, k, k + 1);
    }

    return { mu, ln_var_clamp };
}

torch::Tensor
vmfnb_vae_tImpl::nb_decode_mu(torch::Tensor z)
{
    auto h = nb_mu_dec->forward(z);
    return torch::exp(torch::log_softmax(h, 1) + mu_bias);
}

std::pair<torch::Tensor, torch::Tensor>
vmfnb_vae_tImpl::nb_encode_nu(torch::Tensor x)
{
    namespace F = torch::nn::functional;
    auto h = F::relu(nb_nu_enc->forward(x));

    auto ln_var_clamp = torch::clamp(nb_nu_repr_lnvar->forward(h), -4., 4.);

    return { nb_nu_repr_mean->forward(h), ln_var_clamp };
}

torch::Tensor
vmfnb_vae_tImpl::nb_decode_nu(torch::Tensor z)
{
    auto ret = torch::exp(nb_nu_dec->forward(z) - nu_bias);
    return torch::clamp(ret, 0., nu_max);
}

torch::Tensor
vmfnb_vae_tImpl::reparameterize(torch::Tensor mu, torch::Tensor lnvar)
{
    if (is_training()) {
        auto sig = lnvar.div(2.0).exp();     //
        auto eps = torch::randn_like(lnvar); //
        return mu + eps.mul(sig);            //
    }
    return mu;
}

vmf_out_t
vmfnb_vae_tImpl::vmf_forward(torch::Tensor x)
{
    namespace F = torch::nn::functional;

    const float vmf_eps = 1e-2 / static_cast<float>(x.size(1));

    auto vmf_mu = F::normalize((ln_vmf_mu.exp() + vmf_eps).mul(L),
                               F::NormalizeFuncOptions().p(2).dim(0));

    auto kappa = torch::clamp(ln_kappa->forward(x),
                              fasterlog(kap_min),
                              fasterlog(kap_max))
                     .exp();

    auto estep = take_vmf_estep(x, vmf_mu, kappa);

    auto vmf_recon =
        torch::mm(estep.latent, torch::transpose(vmf_mu, 0, 1)) * filter;
    // auto vmf_recon = F::normalize(r_, F::NormalizeFuncOptions().p(2).dim(1));

    return { vmf_mu, estep.logits, estep.latent, vmf_recon, kappa, dd };
}

vmfnb_vae_out_t
vmfnb_vae_tImpl::forward(torch::Tensor x)
{
    namespace F = torch::nn::functional;

    ///////////////
    // vMF model //
    ///////////////

    auto vmf_ = vmf_forward(x);

    auto vmf_mu = vmf_.mu;
    auto kappa = vmf_.kappa;     //
    auto z = vmf_.latent;        // n x K
    auto vmf_recon = vmf_.recon; // n x D

    /////////////////////////////
    // Negative binomial model //
    /////////////////////////////

    auto nb_mu_enc_out = nb_encode_mu(x, z);
    auto nb_mu_mean = nb_mu_enc_out.first;
    auto nb_mu_lnvar = nb_mu_enc_out.second;
    auto nb_mu_ = nb_decode_mu(reparameterize(nb_mu_mean, nb_mu_lnvar));

    auto nb_nu_enc_out = nb_encode_nu(x);
    auto nb_nu_mean = nb_nu_enc_out.first;
    auto nb_nu_lnvar = nb_nu_enc_out.second;
    auto nb_nu_ = nb_decode_nu(reparameterize(nb_nu_mean, nb_nu_lnvar));

    auto d_ = torch::softplus(depth->forward(x));

    return { nb_mu_,      // reconstruction of the mean
             nb_nu_,      // reconstruction of the overdisp
             d_,          // reconstruction of the depth
             nb_mu_mean,  //
             nb_mu_lnvar, //
             nb_nu_mean,  //
             nb_nu_lnvar, //
             vmf_recon,   // reconstruction of the vMF
             vmf_.logits, // logits for vMF z
             kappa,       //
             vmf_.dd };
}

/// Von Mises-Fisher loss
/// @param x observed data
/// @param y model forward output
torch::Tensor
vmf_loss(torch::Tensor x, vmfnb_vae_out_t yhat)
{
    namespace F = torch::nn::functional;
    const float dd = yhat.dd;
    const float eps = 1e-2 / dd;
    auto yobs = F::normalize(F::relu(x).log1p() + eps,
                             F::NormalizeFuncOptions().p(2).dim(1));

    const float df = std::max(0.5 * dd - 1., 0.);

    auto recon = yhat.vmf_recon; // N x D
    auto kappa = yhat.vmf_kappa; // scalar

    auto llik = torch::sum(yobs * recon, 1) * kappa;
    llik += df * torch::log(kappa) - lbessel(kappa, df);
    llik -= 0.5 * dd * fasterlog(2. * M_PI);

    return -llik.sum();
}

/// Negative Binomial Loss
/// @param x input tensor
/// @param y output
torch::Tensor
nb_loss(torch::Tensor x, vmfnb_vae_out_t y)
{
    const float eps = 1e-4;

    namespace F = torch::nn::functional;

    auto nu = y.nb_recon_nu + eps;
    auto mu = y.nb_recon_mu * y.nb_recon_depth + eps;

    // log-gamma ratio
    auto lg = torch::lgamma(nu) + torch::lgamma(x + 1.);
    lg -= torch::lgamma(nu + x);

    // logit-like llik
    auto denom = torch::log(mu + nu);
    auto pr = x.mul(denom - torch::log(mu));
    pr += nu.mul(denom - torch::log(nu));

    return torch::sum(lg + pr);
}

torch::Tensor
vmfnb_vae_tImpl::normalize_nb_x(torch::Tensor x)
{
    const float eps = 1e-2;

    namespace F = torch::nn::functional;
    auto xn = F::normalize(x.log1p(), F::NormalizeFuncOptions().p(2).dim(1));
    auto xn_std =
        torch::div(torch::sub(xn, x_mean), F::softplus(ln_x_sd) + eps);
    return xn_std;
}

/// @param x
torch::Tensor
vmfnb_vae_tImpl::normalize_vmf_x(torch::Tensor x)
{
    const float eps = 1e-2 / static_cast<float>(x.size(1));

    namespace F = torch::nn::functional;
    auto opt_ = F::NormalizeFuncOptions().p(2).dim(1);

    return F::normalize((x.log1p() + eps).mul(filter), opt_);
}

vmf_estep_out_t
vmfnb_vae_tImpl::take_vmf_estep(torch::Tensor x,
                                torch::Tensor mu,
                                torch::Tensor kappa)
{
    namespace F = torch::nn::functional;
    auto xn = normalize_vmf_x(x);
    auto logits = torch::log_softmax(torch::mm(xn, mu) * kappa, 1);
    if (is_training()) {
        return { logits, logits.exp() };
    }

    auto opt = F::GumbelSoftmaxFuncOptions().hard(true).dim(1);
    auto zz = F::gumbel_softmax(logits, opt);

    return { logits, zz };
}

/// KL-divergence loss from the uniform prior
/// Eq[ln q - ln p]
/// @param ln_q_y variational log probability
torch::Tensor
kl_loss_uniform(torch::Tensor ln_q_)
{
    const float K = ln_q_.size(1);
    return torch::sum(ln_q_.exp() * (ln_q_ + fasterlog(K)), 1).sum();
}

/// KL-divergence loss from the standard Normal
/// Eq[ln q(z|μ, σ) - ln p(z|μ, σ)]
/// @param _mean μ
/// @param _lnvar 2*ln(σ)
torch::Tensor
kl_loss_gaussian(torch::Tensor _mean, torch::Tensor _lnvar)
{
    return -0.5 * torch::sum(1 + _lnvar - _mean.pow(2) - _lnvar.exp());
}

//////////////////
// helper class //
//////////////////

struct vmfnb_recorder_t {

    using Mat =
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    explicit vmfnb_recorder_t(const std::string _hdr, const int64_t _max_epoch)
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

        //////////////////////////
        // output NB mu encoder //
        //////////////////////////

        auto x = db.torch_tensor().to(torch::kCPU);
        auto vmf_ = model->vmf_forward(x);
        auto mu_out = model->nb_encode_mu(x, vmf_.latent);

        torch::Tensor _mean = std::get<0>(mu_out);
        torch::Tensor _lnvar = std::get<1>(mu_out);
        torch::Tensor _clust = vmf_.latent;

        if (mean_out.rows() < ntot || mean_out.cols() < _mean.size(1)) {
            mean_out.resize(ntot, _mean.size(1));
            mean_out.setZero();
        }

        if (lnvar_out.rows() < ntot || lnvar_out.cols() < _lnvar.size(1)) {
            lnvar_out.resize(ntot, _lnvar.size(1));
            lnvar_out.setZero();
        }

        if (clust_out.rows() < ntot || clust_out.cols() < _clust.size(1)) {
            clust_out.resize(ntot, _clust.size(1));
            clust_out.setZero();
        }

        Eigen::Map<Mat> _mean_mat(_mean.data_ptr<float>(),
                                  _mean.size(0),
                                  _mean.size(1));

        Eigen::Map<Mat> _lnvar_mat(_lnvar.data_ptr<float>(),
                                   _lnvar.size(0),
                                   _lnvar.size(1));

        Eigen::Map<Mat> _clust_mat(_clust.data_ptr<float>(),
                                   _clust.size(0),
                                   _clust.size(1));

        for (int64_t j = 0; j < batches.size(); ++j) {
            if (batches[j] < ntot) {
                mean_out.row(batches[j]) = _mean_mat.row(j);
                lnvar_out.row(batches[j]) = _lnvar_mat.row(j);
                clust_out.row(batches[j]) = _clust_mat.row(j);
            }
        }

        model->train(true); // release
    }

    void write(const std::string tag)
    {
        const std::string _hdr_tag =
            tag.size() > 0 ? (header + "_" + tag) : header;
        write_data_file(_hdr_tag + ".mu_mean.gz", mean_out);
        write_data_file(_hdr_tag + ".mu_lnvar.gz", lnvar_out);
        write_data_file(_hdr_tag + ".clust.gz", clust_out);
    }

    const std::string header; // file header
    int64_t epoch;
    const int64_t max_epoch;
    Mat mean_out, lnvar_out, clust_out;
};

struct composite_loss_t {

    composite_loss_t()
    {
        time_discount = 0.1;
        min_rate = 0.01;
        max_rate = 1.;
        max_temp = 50.;
        min_temp = .05;
    }

    /// @param x observed data
    /// @param y reconstruction
    /// @param epoch current epoch
    template <typename X, typename Y>
    torch::Tensor operator()(X x, Y y, const int64_t epoch)
    {
        const float n = x.size(0);

        using namespace mmvae::vmfnb;
        float t = static_cast<float>(epoch);
        float rate = max_rate * std::exp(-time_discount * t);

        auto kl_nb = kl_loss_gaussian(y.nb_mu_mean, y.nb_mu_lnvar) +
            kl_loss_gaussian(y.nb_nu_mean, y.nb_nu_lnvar);

        auto kl_vmf = kl_loss_uniform(y.vmf_logits);

        return (nb_loss(x, y) + vmf_loss(x, y) + rate * (kl_nb + kl_vmf)) / n;
    }

    float time_discount;
    float max_rate;
    float min_rate;
    float max_temp;
    float min_temp;
};

TORCH_MODULE(vmfnb_vae_t); // expose vmfnb_vae_t

} //
} //
#endif
