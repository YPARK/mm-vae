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

template <typename MODEL_PTR, typename RECORDER, typename DATA_BLOCK>
void
visit_vae_model(MODEL_PTR model, RECORDER &recorder, DATA_BLOCK &data_block)
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
    model->to(device);
    model->train(false);

    for (int64_t b = 0; b < nbatch; ++b) {

        const int64_t lb = b * data_block.size();
        const int64_t ub = (b + 1) * data_block.size();

        for (int64_t j = 0; j < (ub - lb); ++j) {
            batch[j] = (lb + j) % ntot;
        }

        data_block.read(batch);
        recorder.update(model, data_block, batch);
        data_block.clear();
    }

    TLOG("Done visit");
}

template <typename MODEL_PTR,
          typename RECORDER,
          typename DATA_BLOCK,
          typename LOSS>
void
train_vae_model(MODEL_PTR model,
                RECORDER &recorder,
                DATA_BLOCK &data_block,
                training_options_t &opt,
                LOSS loss_fun)
{
    TLOG("Training on " << (opt.device.type() == torch::kCUDA ? "GPU" : "CPU"));

    const int64_t ntot = data_block.ntot();
    const int64_t batch_size = data_block.size();
    int64_t nbatch = ntot / data_block.size();
    if ((nbatch * data_block.size()) < ntot) {
        ++nbatch;
    }

    std::vector<typename DATA_BLOCK::Index> batch(data_block.size());

    TLOG("Batch size = " << data_block.size() << ", "
                         << "Number of batches = " << nbatch);

    using optim_t = torch::optim::Adam;
    optim_t adam(model->parameters(), torch::optim::AdamOptions(opt.lr));
    model->to(opt.device);
    model->pretty_print(std::cerr);

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<> rboot(0, data_block.size() - 1);
    std::vector<int64_t> ridx(data_block.size());

    auto _long_opt = torch::TensorOptions().dtype(torch::kLong);

    torch::Tensor _loss(torch::zeros({ 1 }));

    for (int64_t epoch = 0; epoch < opt.max_epoch; ++epoch) {
        const float t = epoch;

        float _loss_epoch = 0.;

        for (int64_t b = 0; b < nbatch; ++b) {

            const int64_t lb = b * data_block.size();
            const int64_t ub = (b + 1) * data_block.size();

            for (int64_t j = 0; j < (ub - lb); ++j) {
                batch[j] = (lb + j) % ntot;
            }

            data_block.read(batch);
            torch::Tensor x = data_block.torch_tensor().to(opt.device);

            model->train(true);

            // Calculate the loss function
            {
                auto y = model->forward(x);
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

                auto yboot = model->forward(xboot);
                auto loss = loss_fun(xboot, yboot, epoch);

                adam.zero_grad();
                loss.backward();
                adam.step();
            }

            if ((epoch + 1) % opt.recording == 0) {
                recorder.update(model, data_block, batch);
            }
            data_block.clear();
        }

        std::cerr << "\r";
        _loss_epoch /= batch_size * nbatch;

        TLOG("t=" << (epoch + 1) << " " << _loss_epoch);

        if ((epoch + 1) % opt.recording == 0) {
            recorder.write(zeropad(epoch + 1, opt.max_epoch));
        }
    }
    TLOG("Done training");
}

#endif
