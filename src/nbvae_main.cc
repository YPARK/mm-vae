#include "mmvae.hh"
#include "mmvae_io.hh"
#include "mmvae_alg.hh"
#include "nb.hh"
#include "vmf.hh"
#include "util.hh"
#include "std_util.hh"
#include "io.hh"
#include <random>
#include <chrono>

struct nbvae_recorder_t {

    using Mat =
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    explicit nbvae_recorder_t(const std::string _hdr)
        : header(_hdr)
    {
    }

    template <typename MODEL, typename DB, typename BAT>
    void update(MODEL &model, DB &db, BAT &batches)
    {
        model->train(false); // freeze the model

        const int64_t ntot = db.ntot();

        ///////////////////////
        // output mu encoder //
        ///////////////////////

        auto mu_out = model->encode_mu(db.torch_tensor().to(torch::kCPU));
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
        write_data_file(_hdr_tag + ".mu_mean.gz", mean_out);
        write_data_file(_hdr_tag + ".mu_lnvar.gz", lnvar_out);
    }

    const std::string header; // file header
    Mat mean_out, lnvar_out;
};

struct nb_loss_t {

    nb_loss_t()
    {
        time_discount = 0.0;
        min_rate = 1e-4;
    }

    /// @param x observed data
    /// @param y reconstruction
    /// @param epoch current epoch
    template <typename X, typename Y>
    torch::Tensor operator()(X x, Y y, const int64_t epoch)
    {
        float t = static_cast<float>(epoch);
        float rate = std::exp(-time_discount * t);
        return mmvae::nb::loss(x, y, std::max(rate, min_rate));
    }

    float time_discount;
    float min_rate;
};

int
main(const int argc, const char *argv[])
{

    mmvae_options_t main_options;
    training_options_t train_opt;
    mmvae::nb::nbvae_options_t vae_opt;

    CHK(parse_mmvae_options(argc, argv, main_options));
    CHK(parse_nbvae_options(argc, argv, vae_opt));
    CHK(parse_training_options(argc, argv, train_opt));

    if (!file_exists(main_options.mtx))
        return EXIT_FAILURE;

    using data_t = mmvae::mtx_data_block_t;

    const auto mtx_file = main_options.mtx;
    const auto idx_file = main_options.idx;

    if (!file_exists(idx_file))
        CHK(mmutil::index::build_mmutil_index(mtx_file, idx_file));

    const int64_t batch_size = main_options.batch_size;

    // data loader from the .mtx file
    data_t data_block(mtx_file, idx_file, batch_size);

    TLOG("Constructing a model");
    std::vector<mmvae::nb::mu_encoder_h_dim> henc;
    std::vector<mmvae::nb::mu_decoder_h_dim> hdec;

    std::transform(std::begin(vae_opt.mean_encoding_layers),
                   std::end(vae_opt.mean_encoding_layers),
                   std::back_inserter(henc),
                   [](const int64_t d) {
                       return mmvae::nb::mu_encoder_h_dim(d);
                   });

    std::transform(std::begin(vae_opt.mean_decoding_layers),
                   std::end(vae_opt.mean_decoding_layers),
                   std::back_inserter(hdec),
                   [](const int64_t d) {
                       return mmvae::nb::mu_decoder_h_dim(d);
                   });

    mmvae::nb::nbvae_t model(mmvae::nb::data_dim(data_block.nfeature()),
                             henc,
                             hdec,
                             mmvae::nb::mu_encoder_r_dim(vae_opt.mean_latent),
                             mmvae::nb::nu_encoder_h_dim(
                                 vae_opt.overdispersion_encoding),
                             mmvae::nb::nu_encoder_r_dim(
                                 vae_opt.overdispersion_latent),
                             false);

    nbvae_recorder_t recorder(main_options.out);

    nb_loss_t loss;
    loss.time_discount = main_options.kl_discount;

    train_vae_model(model, recorder, data_block, train_opt, loss);
    TLOG("Writing down results...");

    visit_vae_model(model, recorder, data_block);
    recorder.write("");
    TLOG("Done");
    return EXIT_SUCCESS;
}
