#ifndef MMVAE_HH_
#define MMVAE_HH_

#include <iostream>
#include "util.hh"
#include "gzstream.hh"

#include "io_visitor.hh"
#include "eigen_util.hh"
#include "std_util.hh"
#include "math.hh"
#include "check.hh"

#include "io.hh"
#include "gzstream.hh"
#include "mmutil_index.hh"
#include "mmutil_bgzf_util.hh"

#ifndef TORCH_INC_H_     // Prevent torch
#define TORCH_INC_H_     // from reloading
#include <torch/torch.h> //
#endif                   // End of torch

#include <getopt.h>

struct mmvae_options_t {

    explicit mmvae_options_t()
    {
        // default options
        batch_size = 100;
        kl_discount = 0.;
        kl_min = 1.;
    }

    std::string mtx;
    std::string idx;
    std::string out;
    int64_t batch_size;
    float kl_discount;
    float kl_min;
};

int
parse_mmvae_options(const int argc,
                    const char *_argv[],
                    mmvae_options_t &options)
{
    const char *_usage =
        "\n"
        "[options]\n"
        "\n"
        "--mtx         : matrix market mtx file\n"
        "--idx         : matrix market mtx index file (default: ${mtx}.index\n"
        "--out         : output file header\n"
        "--batch_size  : #samples in each batch (default: 100)\n"
        "\n"
        "--kl_discount : KL divergence discount (default: 0)\n"
        "              : Loss = likelihood_loss + beta * KL_loss\n"
        "              : where beta = exp(- ${discount} * epoch)\n"
        "--kl_min      : min KL divergence penalty (default: 1)\n"
        "\n";

    const char *const short_opts = "M:I:O:b:K:l:h?";

    const option long_opts[] = {
        { "mtx", required_argument, nullptr, 'M' },         //
        { "idx", required_argument, nullptr, 'I' },         //
        { "out", required_argument, nullptr, 'O' },         //
        { "output", required_argument, nullptr, 'O' },      //
        { "batch_size", required_argument, nullptr, 'b' },  //
        { "batch", required_argument, nullptr, 'b' },       //
        { "kl_discount", required_argument, nullptr, 'K' }, //
        { "kl_min", required_argument, nullptr, 'l' },      //
        { "help", no_argument, nullptr, 'h' },              //
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
        case 'M':
            options.mtx = std::string(optarg);
            break;

        case 'I':
            options.idx = std::string(optarg);
            break;

        case 'O':
            options.out = std::string(optarg);
            break;

        case 'b':
            options.batch_size = std::stol(optarg);
            break;

        case 'K':
            options.kl_discount = std::stof(optarg);
            break;

        case 'l':
            options.kl_min = std::stof(optarg);
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

    ERR_RET(!file_exists(options.mtx), "missing mtx file");
    ERR_RET(options.out.size() == 0, "need output file header");

    if (options.idx.size() == 0) {
        options.idx = options.mtx + ".index";
    }

    return EXIT_SUCCESS;
}

#endif
