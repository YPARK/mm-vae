#include "mmvae.hh"
#include "mmvae_io.hh"
#include "mmvae_alg.hh"
#include "vmfnb.hh"
#include "util.hh"
#include "std_util.hh"
#include "io.hh"

int
main(const int argc, const char *argv[])
{

    using namespace mmvae::vmfnb;

    mmvae_options_t main_opt;
    training_options_t train_opt;
    vmfnb_options_t vmfnb_opt;

    CHK(parse_mmvae_options(argc, argv, main_opt));
    CHK(parse_vmfnb_options(argc, argv, vmfnb_opt));
    CHK(parse_training_options(argc, argv, train_opt));

    if (!file_exists(main_opt.mtx))
        return EXIT_FAILURE;

    const auto mtx_file = main_opt.mtx;
    const auto idx_file = main_opt.idx;

    const auto feature_file = main_opt.row;
    const auto annot_file = main_opt.annot;

    if (!file_exists(idx_file))
        CHK(mmutil::index::build_mmutil_index(mtx_file, idx_file));

    // // data loader from the .mtx file
    // const int64_t batch_size = main_opt.batch_size;
    // mmvae::mtx_data_block_t data_block(mtx_file, idx_file, batch_size);

    // std::vector<mu_shared_enc_h_dim> henc;
    // std::vector<mu_nb_dec_h_dim> hdec;
    // std::vector<mu_vmf_dec_h_dim> vdec;

    // std::transform(std::begin(vmfnb_opt.mean_encoding_layers),
    //                std::end(vmfnb_opt.mean_encoding_layers),
    //                std::back_inserter(henc),
    //                [](const int64_t d) { return mu_shared_enc_h_dim(d); });

    // std::transform(std::begin(vmfnb_opt.mean_decoding_layers),
    //                std::end(vmfnb_opt.mean_decoding_layers),
    //                std::back_inserter(hdec),
    //                [](const int64_t d) { return mu_nb_dec_h_dim(d); });

    // std::transform(std::begin(vmfnb_opt.vmf_decoding_layers),
    //                std::end(vmfnb_opt.vmf_decoding_layers),
    //                std::back_inserter(vdec),
    //                [](const int64_t d) { return mu_vmf_dec_h_dim(d); });

    // TLOG("Constructing a model");

    // vmfnb_vae_t model(data_dim(data_block.nfeature()),
    //                   henc,
    //                   hdec,
    //                   mu_shared_enc_r_dim(vmfnb_opt.mean_latent),
    //                   nu_nb_enc_h_dim(vmfnb_opt.overdispersion_encoding),
    //                   nu_nb_enc_r_dim(vmfnb_opt.overdispersion_latent),
    //                   vdec,
    //                   kappa_min(vmfnb_opt.kappa_min),
    //                   kappa_max(vmfnb_opt.kappa_max),
    //                   vmfnb_opt.do_relu);

    // vmfnb_recorder_t recorder(main_opt.out, train_opt.max_epoch);

    // composite_loss_t loss;
    // loss.time_discount = main_opt.kl_discount;
    // loss.max_rate = main_opt.kl_max;
    // loss.min_rate = main_opt.kl_min;

    // train_vae_model(model, recorder, data_block, train_opt, loss);

    TLOG("Done");
    return EXIT_SUCCESS;
}
