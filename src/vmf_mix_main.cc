#include "mmvae.hh"
#include "mmvae_io.hh"
#include "mmvae_alg.hh"
#include "vmf.hh"
#include "util.hh"
#include "std_util.hh"
#include "io.hh"

#include <random>
#include <chrono>

struct vmf_visitor_t {

    using Mat =
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    using IntMat =
        Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    explicit vmf_visitor_t(const std::string _hdr, torch::Tensor _L)
        : header(_hdr)
        , L(_L)
    {
    }

    template <typename MODEL, typename DB, typename BAT>
    void update_on_batch(MODEL &model, DB &db, BAT &batches)
    {
        const int64_t ntot = db.ntot();
        const int64_t D = model->D;
        const int64_t K = model->K;

        if (membership.size() < ntot)
            membership.resize(ntot);

        model->train(false); // freeze the model

        torch::Tensor argmax =
            torch::argmax(model->take_estep(db.torch_tensor().to(torch::kCPU)),
                          1);

        std::vector<int64_t> _temp(argmax.data_ptr<int64_t>(),
                                   argmax.data_ptr<int64_t>() + argmax.numel());

        for (int64_t j = 0; j < batches.size(); ++j) {
            const int64_t i = batches[j];
            const int64_t k = _temp[j];
            membership[i] = k;
        }

        model->train(true); // release
    }

    template <typename MODEL, typename DB, typename BAT>
    void update_on_epoch(MODEL &model, DB &db, BAT &batches)
    {
        TLOG(model->kappa);

        std::vector<int64_t> count(model->K);
        std::fill(count.begin(), count.end(), 0.);
        for (auto k : membership) {
            count[k]++;
        }
        for (int k = 0; k < model->K; ++k) {
            std::cerr << count[k] << " ";
        }
        std::cerr << std::endl;
    }

    void write(const std::string tag) { }

    const std::string header; // file header

    torch::Tensor L; // D x K constraint matrix

    std::vector<int64_t> membership;

    // torch::Tensor xsum; // D x K sufficient stat
    // torch::Tensor nsum; // 1 x K sufficient stat
    // torch::Tensor z;    // n x K latent membership
};

struct vmf_loss_t {

    vmf_loss_t()
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
        using namespace mmvae::vmf;
        float t = static_cast<float>(epoch);
        float rate = max_rate * std::exp(-time_discount * t);
        // float temp = max_temp * std::exp(-time_discount * t); // cool down
        auto nllik = std::get<0>(y);
        auto ln_q = std::get<1>(y);
        const float n = nllik.size(0);
        return torch::sum(nllik) / n + rate * kl_loss_uniform(ln_q).sum() / n;
    }

    float time_discount;
    float max_rate;
    float min_rate;
    float max_temp;
    float min_temp;
};

int
main(const int argc, const char *argv[])
{

    using namespace mmvae::vmf;

    mmvae_options_t main_options;
    training_options_t train_opt;
    vmf_options_t vmf_options;

    CHK(parse_mmvae_options(argc, argv, main_options));
    CHK(parse_vmf_options(argc, argv, vmf_options));
    CHK(parse_training_options(argc, argv, train_opt));

    if (!file_exists(main_options.mtx))
        return EXIT_FAILURE;

    const auto mtx_file = main_options.mtx;
    const auto idx_file = main_options.idx;

    const auto feature_file = main_options.row;
    const auto annot_file = main_options.annot;

    if (!file_exists(idx_file))
        CHK(mmutil::index::build_mmutil_index(mtx_file, idx_file));

    if (!file_exists(annot_file) || !file_exists(feature_file))
        return EXIT_FAILURE;

    // data loader from the .mtx file
    const int64_t batch_size = main_options.batch_size;
    mmvae::mtx_data_block_t data_block(mtx_file, idx_file, batch_size);

    /////////////////////////////
    // build annotation matrix //
    /////////////////////////////

    std::vector<std::tuple<std::string, std::string>> _annot;
    std::vector<std::string> features;

    CHK(read_pair_file(annot_file, _annot));
    CHK(read_vector_file(feature_file, features));

    auto feature2id = make_position_dict<std::string, int64_t>(features);
    std::unordered_map<std::string, int64_t> label_pos;
    std::vector<std::string> labels;

    {
        int64_t j = 0;
        for (auto pp : _annot) {
            if (feature2id.count(std::get<0>(pp)) > 0) {
                if (label_pos.count(std::get<1>(pp)) == 0) {
                    label_pos[std::get<1>(pp)] = j++;
                    labels.push_back(std::get<1>(pp));
                }
            }
        }
    }

    const int64_t D = feature2id.size();
    const int64_t K = std::max(label_pos.size(), (std::size_t)1);

    torch::Tensor L = torch::zeros({ D, K });

    for (auto pp : _annot) {
        if (feature2id.count(std::get<0>(pp)) > 0) {
            const int64_t k = label_pos[std::get<1>(pp)];
            const int64_t j = feature2id[std::get<0>(pp)];
            L[j][k] = 1.;
        }
    }

    for (auto s : labels) {
        std::cerr << s << " ";
    }
    std::cerr << std::endl;

    ///////////////////
    // build a model //
    ///////////////////

    mmvae::vmf::vmf_mixture_t model(L,
                                    kappa_min(vmf_options.kappa_min),
                                    kappa_max(vmf_options.kappa_max));

    TLOG("Constructing a model");

    vmf_visitor_t recorder(main_options.out, L);

    vmf_loss_t loss;
    train_vae_model(model, recorder, data_block, train_opt, loss);

    TLOG("Done");
    return EXIT_SUCCESS;
}
