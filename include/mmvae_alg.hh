#include "mmvae.hh"
#include "mmvae_io.hh"
#include "util.hh"
#include "std_util.hh"
#include "io.hh"
#include <random>
#include <chrono>

#include <getopt.h>

#ifndef MMVAE_ALG_HH_
#define MMVAE_ALG_HH_

struct training_options_t {

    explicit training_options_t()
        : device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
    {
        lr = 1e-3;
        nboot = 3;
        max_epoch = 101;
        recording = 10;
    }

    float lr;          // 1e-3;
    int64_t nboot;     // 3;
    int64_t max_epoch; // 31;
    int64_t recording; // 3;

    torch::Device device;
};

int
parse_training_options(const int argc,
                       const char *_argv[],
                       training_options_t &options)
{
    const char *_usage = "[Training algorithm options]\n"
                         "\n"
                         "--lr         : learning rate (default: 1e-3)\n"
                         "--nboot      : #bootstrapped gradients (default: 3)\n"
                         "--max_epoch  : maximum #epoch (default: 101)\n"
                         "--recording  : recording interval (default: 10)\n"
                         "\n";

    const char *const short_opts = "L:B:E:R:h";

    const option long_opts[] = {
        { "lr", required_argument, nullptr, 'L' },            //
        { "learning", required_argument, nullptr, 'L' },      //
        { "learn_rate", required_argument, nullptr, 'L' },    //
        { "learning_rate", required_argument, nullptr, 'L' }, //
        { "nboot", required_argument, nullptr, 'B' },         //
        { "boot", required_argument, nullptr, 'B' },          //
        { "bootstrap", required_argument, nullptr, 'B' },     //
        { "max_epoch", required_argument, nullptr, 'E' },     //
        { "epoch", required_argument, nullptr, 'E' },         //
        { "recording", required_argument, nullptr, 'R' },     //
        { "help", no_argument, nullptr, 'h' },                //
        { nullptr, no_argument, nullptr, 0 }
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
        case 'L':
            options.lr = std::stof(optarg);
            break;

        case 'B':
            options.nboot = std::stol(optarg);
            break;

        case 'E':
            options.max_epoch = std::stol(optarg);
            break;

        case 'R':
            options.recording = std::stol(optarg);
            break;

        case 'h': // -h or --help
            std::cerr << _usage << std::endl;
            for (std::size_t i = 0; i < argv_copy.size(); i++)
                delete[] argv_copy[i];

            return EXIT_SUCCESS;

        case '?': // Unrecognized option
        default:  //
                 ;
        }
    }

    for (std::size_t i = 0; i < argv_copy.size(); i++)
        delete[] argv_copy[i];

    return EXIT_SUCCESS;
}

template <typename VISITOR, typename DATA_BLOCK>
void
visit_data(VISITOR &visitor, DATA_BLOCK &data_block)
{
    const int64_t ntot = data_block.ntot();
    const int64_t batch_size = data_block.size();
    int64_t nbatch = ntot / data_block.size();
    if ((nbatch * data_block.size()) < ntot) {
        ++nbatch;
    }

    std::vector<typename DATA_BLOCK::Index> batch(data_block.size());

    TLOG("Batch size = " << data_block.size() << ", "
                         << "Number of batches = " << nbatch);

    torch::Device device(torch::kCPU);

    for (int64_t b = 0; b < nbatch; ++b) {

        const int64_t lb = b * data_block.size();
        const int64_t ub = (b + 1) * data_block.size();

        for (int64_t j = 0; j < (ub - lb); ++j) {
            batch[j] = (lb + j) % ntot;
        }

        data_block.read(batch);
        visitor.update_on_batch(data_block, batch);
        data_block.clear();
    }

    TLOG("Done visit");
}

template <typename MODEL_PTR, typename VISITOR, typename DATA_BLOCK>
void
visit_vae_model(MODEL_PTR model, VISITOR &visitor, DATA_BLOCK &data_block)
{
    const int64_t ntot = data_block.ntot();
    const int64_t batch_size = data_block.size();
    int64_t nbatch = ntot / batch_size;
    if ((nbatch * batch_size) < ntot) {
        ++nbatch;
    }

    std::vector<typename DATA_BLOCK::Index> batch(batch_size);

    TLOG("Batch size = " << batch_size << ", "
                         << "Number of batches = " << nbatch);

    torch::Device device(torch::kCPU);
    model->to(device);

    for (int64_t b = 0; b < nbatch; ++b) {

        const int64_t lb = b * batch_size;
        const int64_t ub = (b + 1) * batch_size;

        for (int64_t j = 0; j < (ub - lb); ++j) {
            batch[j] = (lb + j) % ntot;
        }

        data_block.read(batch);
        visitor.update_on_batch(model, data_block, batch);
        data_block.clear();
    }

    // visitor.update_on_epoch(model, data_block, batch);

    TLOG("Done visit");
}

template <typename MODEL_PTR,
          typename VISITOR,
          typename DATA_BLOCK,
          typename LOSS>
void
train_vae_model(MODEL_PTR model,
                VISITOR &visitor,
                DATA_BLOCK &data_block,
                DATA_BLOCK &covar_block,
                training_options_t &opt,
                LOSS loss_fun)
{
    TLOG("Training on " << (opt.device.type() == torch::kCUDA ? "GPU" : "CPU"));

    const int64_t ntot = data_block.ntot();

    ASSERT(ntot == covar_block.ntot(),
           "data and covar on the same set of data points");

    const int64_t batch_size = data_block.size();

    ASSERT(batch_size == covar_block.size(),
           "data and covar on the same batch size");

    int64_t nbatch = ntot / batch_size;
    if ((nbatch * batch_size) < ntot) {
        ++nbatch;
    }

    std::vector<typename DATA_BLOCK::Index> batch(batch_size);

    TLOG("Batch size = " << batch_size << ", "
                         << "Number of batches = " << nbatch);

    using optim_t = torch::optim::Adam;
    optim_t adam(model->parameters(),
                 torch::optim::AdamOptions(opt.lr).weight_decay(1e-4));
    model->to(opt.device);
    model->pretty_print(std::cerr);

    const float grad_clip = 1e-2;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<> rboot(0, batch_size - 1);
    std::vector<int64_t> ridx(batch_size);

    auto _long_opt = torch::TensorOptions().dtype(torch::kLong);

    torch::Tensor _loss(torch::zeros({ 1 }));

    for (int64_t epoch = 0; epoch < opt.max_epoch; ++epoch) {
        const float t = epoch;

        float _loss_epoch = 0.;

        for (int64_t b = 0; b < nbatch; ++b) {

            const int64_t lb = b * batch_size;
            const int64_t ub = (b + 1) * batch_size;

            for (int64_t j = 0; j < (ub - lb); ++j) {
                batch[j] = (lb + j) % ntot;
            }

            data_block.read(batch);
            covar_block.read(batch);

            torch::Tensor x = data_block.torch_tensor().to(opt.device);
            torch::Tensor c = covar_block.torch_tensor().to(opt.device);

            model->train(true);

            // Calculate the loss function
            {
                auto y = model->forward(x, c);
                auto loss = loss_fun(x, y, epoch);
                _loss = loss.sum();
                float _loss_batch = _loss.item<float>();
                _loss_epoch += _loss_batch * batch_size;
                std::cerr << "\r[" << std::setw(20) << (b + 1) << "] ";
                std::cerr << std::setw(20) << _loss_batch;
            }

            // Update gradients many times by sampling with
            // replacement

            for (int64_t boot = 0; boot < opt.nboot; ++boot) {

                for (int64_t r = 0; r < ridx.size(); ++r)
                    ridx[r] = rboot(rng);

                torch::Tensor _ridx =
                    torch::from_blob(ridx.data(),
                                     { static_cast<long long>(ridx.size()) },
                                     _long_opt);

                torch::Tensor xboot = torch::index_select(x, 0, _ridx);
                torch::Tensor cboot = torch::index_select(c, 0, _ridx);

                auto yboot = model->forward(xboot, cboot);
                auto loss = loss_fun(xboot, yboot, epoch);

                adam.zero_grad();
                loss.backward();
                torch::nn::utils::clip_grad_norm_(model->parameters(),
                                                  grad_clip);
                adam.step();
            }

            model->train(false);

            if ((epoch + 1) % opt.recording == 0) {
                visitor.update_on_batch(model, data_block, batch);
            }

            data_block.clear();
            covar_block.clear();
        }

        std::cerr << "\r";
        _loss_epoch /= batch_size * nbatch;

        TLOG("[" << std::setw(20) << (epoch + 1) << "] " << std::setw(20)
                 << _loss_epoch);

        if ((epoch + 1) % opt.recording == 0) {
            visitor.update_on_epoch(model, data_block, batch);
        }
    }
    TLOG("Done training");
}

#endif
