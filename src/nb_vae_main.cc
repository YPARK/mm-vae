#include "mmvae.hh"
#include "mmvae_io.hh"
#include "mmvae_alg.hh"
#include "nb.hh"
#include "util.hh"
#include "std_util.hh"
#include "io.hh"
#include <random>
#include <chrono>

// More customized loss function
struct nb_loss_t {

    nb_loss_t()
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
        return mmvae::nb::loss(x, y, std::max(rate, min_rate));
    }

    float time_discount;
    float max_rate;
    float min_rate;
};

int
main(const int argc, const char *argv[])
{

    mmvae_options_t main_options;
    training_options_t train_opt;
    mmvae::nb::nbvae_options_t nb_opt;

    CHK(parse_mmvae_options(argc, argv, main_options));
    CHK(parse_nbvae_options(argc, argv, nb_opt));
    CHK(parse_training_options(argc, argv, train_opt));

    if (!file_exists(main_options.mtx))
        return EXIT_FAILURE;

    const auto mtx_file = main_options.mtx;
    const auto idx_file = main_options.idx;

    if (!file_exists(idx_file))
        CHK(mmutil::index::build_mmutil_index(mtx_file, idx_file));

    const int64_t batch_size = main_options.batch_size;

    // data loader from the .mtx file
    mmvae::mtx_data_block_t data_block(mtx_file, idx_file, batch_size);

    TLOG("Constructing a model");
    std::vector<mmvae::nb::mu_encoder_h_dim> henc;
    std::vector<mmvae::nb::mu_decoder_h_dim> hdec;

    std::transform(std::begin(nb_opt.mean_encoding_layers),
                   std::end(nb_opt.mean_encoding_layers),
                   std::back_inserter(henc),
                   [](const int64_t d) {
                       return mmvae::nb::mu_encoder_h_dim(d);
                   });

    std::transform(std::begin(nb_opt.mean_decoding_layers),
                   std::end(nb_opt.mean_decoding_layers),
                   std::back_inserter(hdec),
                   [](const int64_t d) {
                       return mmvae::nb::mu_decoder_h_dim(d);
                   });

    mmvae::nb::nbvae_t model(mmvae::nb::data_dim(data_block.nfeature()),
                             henc,
                             hdec,
                             mmvae::nb::mu_encoder_r_dim(nb_opt.mean_latent),
                             mmvae::nb::nu_encoder_h_dim(
                                 nb_opt.overdispersion_encoding),
                             mmvae::nb::nu_encoder_r_dim(
                                 nb_opt.overdispersion_latent),
                             nb_opt.do_relu);

    mmvae::nb::nbvae_recorder_t recorder(main_options.out, train_opt.max_epoch);

    nb_loss_t loss;
    loss.min_rate = main_options.kl_min;
    loss.max_rate = main_options.kl_max;
    loss.time_discount = main_options.kl_discount;

    train_vae_model(model, recorder, data_block, train_opt, loss);
    TLOG("Writing down results...");

    visit_vae_model(model, recorder, data_block);
    recorder.write("");
    TLOG("Done");
    return EXIT_SUCCESS;
}
