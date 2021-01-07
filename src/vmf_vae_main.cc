#include "mmvae.hh"
#include "mmvae_io.hh"
#include "mmvae_alg.hh"
#include "vmf.hh"
#include "util.hh"
#include "std_util.hh"
#include "io.hh"

#include <random>
#include <chrono>

// More customized loss function
struct vmf_loss_t {

    vmf_loss_t()
    {
        time_discount = 0.0;
        min_rate = 1;
        max_rate = 1;
    }

    /// @param x observed data
    /// @param y reconstruction
    /// @param epoch current epoch
    template <typename X, typename Y>
    torch::Tensor operator()(X x, Y y, const int64_t epoch)
    {
        float t = static_cast<float>(epoch);
        float rate = max_rate * std::exp(-time_discount * t);
        return mmvae::vmf::vmf_vae_loss(x, y, std::max(rate, min_rate));
    }

    float time_discount;
    float max_rate;
    float min_rate;
};

int
main(const int argc, const char *argv[])
{

    using namespace mmvae::vmf;

    mmvae_options_t main_options;
    training_options_t train_opt;
    vmf_options_t vmf_opt;

    CHK(parse_mmvae_options(argc, argv, main_options));
    CHK(parse_vmf_options(argc, argv, vmf_opt));
    CHK(parse_training_options(argc, argv, train_opt));

    if (!file_exists(main_options.mtx))
        return EXIT_FAILURE;

    const auto mtx_file = main_options.mtx;
    const auto idx_file = main_options.idx;

    if (!file_exists(idx_file))
        CHK(mmutil::index::build_mmutil_index(mtx_file, idx_file));

    // data loader from the .mtx file
    const int64_t batch_size = main_options.batch_size;
    mmvae::mtx_data_block_t data_block(mtx_file, idx_file, batch_size);

    // create another data loader for the covariates
    auto covar_mtx_file = main_options.covar_mtx;
    auto covar_idx_file = main_options.covar_idx;

    if (!file_exists(covar_mtx_file)) {
        covar_mtx_file = main_options.out + ".covar.mtx.gz";
        covar_idx_file = covar_mtx_file + ".index";
        create_ones_like(data_block, covar_mtx_file);
        TLOG("No covariate file is given. So we use this: " << covar_mtx_file);
        CHK(mmutil::index::build_mmutil_index(covar_mtx_file, covar_idx_file));
    } else {
        if (!file_exists(covar_idx_file))
            CHK(mmutil::index::build_mmutil_index(covar_mtx_file,
                                                  covar_idx_file));
    }

    mmvae::mtx_data_block_t covar_block(covar_mtx_file,
                                        covar_idx_file,
                                        batch_size);

    TLOG("Constructing a model");

    std::vector<mmvae::vmf::z_enc_dim> henc;
    std::vector<mmvae::vmf::z_dec_dim> hdec;

    std::transform(std::begin(vmf_opt.encoding_layers),
                   std::end(vmf_opt.encoding_layers),
                   std::back_inserter(henc),
                   [](const int64_t d) { return mmvae::vmf::z_enc_dim(d); });

    std::transform(std::begin(vmf_opt.decoding_layers),
                   std::end(vmf_opt.decoding_layers),
                   std::back_inserter(hdec),
                   [](const int64_t d) { return mmvae::vmf::z_dec_dim(d); });

    mmvae::vmf::vmf_vae_t model(mmvae::vmf::data_dim(data_block.nfeature()),
                                mmvae::vmf::covar_dim(covar_block.nfeature()),
                                mmvae::vmf::z_repr_dim(vmf_opt.latent),
                                henc,
                                hdec,
                                mmvae::vmf::kappa_min(vmf_opt.kappa_min),
                                mmvae::vmf::kappa_max(vmf_opt.kappa_max),
                                vmf_opt.do_relu);

    vmf_vae_recorder_t recorder(main_options.out, train_opt.max_epoch);

    vmf_loss_t loss;
    loss.min_rate = main_options.kl_min;
    loss.max_rate = main_options.kl_max;
    loss.time_discount = main_options.kl_discount;

    train_vae_model(model, recorder, data_block, covar_block, train_opt, loss);

    TLOG("Done");
    return EXIT_SUCCESS;
}
