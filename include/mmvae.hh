#ifndef MMVAE_HH_
#define MMVAE_HH_

#include <iostream>
#include "util.hh"
#include "gzstream.hh"

#include "io_visitor.hh"
#include "io_alg.hh"
#include "eigen_util.hh"
#include "std_util.hh"
#include "math.hh"
#include "check.hh"

#include "io.hh"
#include "gzstream.hh"
#include "mmutil_index.hh"
#include "mmutil_bgzf_util.hh"

#ifndef TORCH_INC_H_                               // Prevent torch
#define TORCH_INC_H_                               // from reloading
#include <torch/torch.h>                           //
#include <torch/csrc/autograd/variable.h>          //
#include <torch/csrc/autograd/function.h>          //
#include <torch/csrc/autograd/VariableTypeUtils.h> //
#include <torch/csrc/autograd/functions/utils.h>   //
#endif                                             // End of torch

#include <getopt.h>

struct mmvae_options_t {

    explicit mmvae_options_t()
    {
        // default options
        batch_size = 100;
        kl_discount = .1;
        kl_min = 1e-2;
        kl_max = 1.;
    }

    std::string mtx;
    std::string idx;
    std::string out;
    std::string row;
    std::string col;
    std::string annot;

    std::string covar_mtx;
    std::string covar_idx;

    int64_t batch_size;
    float kl_discount;
    float kl_min;
    float kl_max;
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
        "--mtx         : a matrix market mtx file\n"
        "--idx         : an index file for the mtx (default: ${mtx}.index)\n"
        "--row         : the rows (line = one string)\n"
        "--col         : the columns (line = one string)\n"
        "--annot       : the list of column annotations (line = ${col} <space> ${k})\n"
        "--out         : output file header\n"
        "--covar       : a separate matrix market mtx for other covariates\n"
        "--covar_idx   : an index file for the covar mtx (default: ${covar}.index)\n"
        "--batch_size  : #samples in each batch (default: 100)\n"
        "\n"
        "--kl_discount : KL divergence discount (default: 0)\n"
        "              : Loss = likelihood_loss + beta * KL_loss\n"
        "              : where beta = exp(- ${discount} * epoch)\n"
        "--kl_max      : max KL divergence penalty (default: 1)\n"
        "--kl_min      : min KL divergence penalty (default: 1e-2)\n"
        "\n";

    const char *const short_opts = "M:I:O:V:J:r:c:a:b:K:L:l:h?";

    const option long_opts[] = {
        { "mtx", required_argument, nullptr, 'M' },         //
        { "idx", required_argument, nullptr, 'I' },         //
        { "out", required_argument, nullptr, 'O' },         //
        { "output", required_argument, nullptr, 'O' },      //
        { "cov", required_argument, nullptr, 'V' },         //
        { "covar", required_argument, nullptr, 'V' },       //
        { "cov_idx", required_argument, nullptr, 'J' },     //
        { "covar_idx", required_argument, nullptr, 'J' },   //
        { "row", required_argument, nullptr, 'r' },         //
        { "col", required_argument, nullptr, 'c' },         //
        { "column", required_argument, nullptr, 'c' },      //
        { "annot", required_argument, nullptr, 'a' },       //
        { "annotation", required_argument, nullptr, 'a' },  //
        { "batch_size", required_argument, nullptr, 'b' },  //
        { "batch", required_argument, nullptr, 'b' },       //
        { "kl_discount", required_argument, nullptr, 'K' }, //
        { "kl_max", required_argument, nullptr, 'L' },      //
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

        case 'V':
            options.covar_mtx = std::string(optarg);
            break;

        case 'J':
            options.covar_idx = std::string(optarg);
            break;

        case 'O':
            options.out = std::string(optarg);
            break;

        case 'r':
            options.row = std::string(optarg);
            break;

        case 'c':
            options.col = std::string(optarg);
            break;

        case 'a':
            options.annot = std::string(optarg);
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

        case 'L':
            options.kl_max = std::stof(optarg);
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

    if (options.covar_idx.size() == 0) {
        options.covar_idx = options.covar_mtx + ".index";
    }

    return EXIT_SUCCESS;
}

struct annotation_t {

    using Str = std::string;

    struct FEAT {
        Str s;
    };

    struct ANNOT {
        Str s;
    };

    explicit annotation_t(const ANNOT annot, const FEAT feat);

    torch::Tensor operator()();

    const std::string annot_file;
    const std::string feature_file;

private:
    std::vector<std::tuple<std::string, std::string>> annot_;
    std::unordered_map<std::string, int64_t> feature2id;
    std::vector<std::string> features;
    std::unordered_map<std::string, int64_t> label_pos;
    std::vector<std::string> labels;

    int64_t D, K;
};

annotation_t::annotation_t(const annotation_t::ANNOT annot,
                           const annotation_t::FEAT feat)
    : annot_file(annot.s)
    , feature_file(feat.s)
{

    CHK(read_pair_file(annot_file, annot_));
    CHK(read_vector_file(feature_file, features));

    feature2id = make_position_dict<std::string, int64_t>(features);

    {
        int64_t j = 0;
        for (auto pp : annot_) {
            if (feature2id.count(std::get<0>(pp)) > 0) {
                if (label_pos.count(std::get<1>(pp)) == 0) {
                    label_pos[std::get<1>(pp)] = j++;
                    labels.push_back(std::get<1>(pp));
                }
            }
        }
    }

    D = feature2id.size();
    K = std::max(label_pos.size(), (std::size_t)1);
}

torch::Tensor
annotation_t::operator()()
{
    torch::Tensor L(torch::zeros({ D, K }));

    for (auto pp : annot_) {
        if (feature2id.count(std::get<0>(pp)) > 0) {
            const int64_t k = label_pos[std::get<1>(pp)];
            const int64_t j = feature2id[std::get<0>(pp)];
            L[j][k] = 1.;
        }
    }

    return L;
}

#endif
