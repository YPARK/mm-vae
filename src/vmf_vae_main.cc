#include "mmvae.hh"
#include "mmvae_io.hh"
#include "mmvae_alg.hh"
#include "vmf.hh"
#include "util.hh"
#include "std_util.hh"
#include "io.hh"

#include <random>
#include <chrono>

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

    train_vae_model(model, recorder, data_block, train_opt, loss);

    TLOG("Done");
    return EXIT_SUCCESS;
}
