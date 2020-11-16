#ifndef MMVAE_HH_
#define MMVAE_HH_

#include <iostream>
#include "util.hh"
#include "gzstream.hh"

#include "io.hh"
#include "io_visitor.hh"
#include "eigen_util.hh"
#include "std_util.hh"
#include "math.hh"
#include "check.hh"
#include "mmutil_index.hh"
#include "gzstream.hh"
#include "mmutil_bgzf_util.hh"

#ifndef TORCH_INC_H_     // Prevent torch
#define TORCH_INC_H_     // from reloading
#include <torch/torch.h> //
#endif                   //

#endif
