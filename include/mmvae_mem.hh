#include "mmvae.hh"
#include "io_alg.hh"
#include "io_visitor.hh"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Sparse>
#include "eigen_util.hh"

#ifndef MMVAE_MEM_HH_
#define MMVAE_MEM_HH_

namespace mmvae {

// Basic idea
// 1. Store everthing in the memory as a sparse matrix
// 2. Access them and return a dense matrix for subsets
struct mtx_memory_block_t {

    using Scalar = float;
    using Index = ptrdiff_t;
    using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, std::ptrdiff_t>;

    explicit mtx_memory_block_t(const std::string _mtx,
                                const std::string _idx,
                                const Index batch_size);

    /// Populate the memory by reading the data from .mtx
    /// @param subcol
    void read(const std::vector<Index> &subcol);
    void clear();

    /// @returns torch tensor (batch_size x feature)
    torch::Tensor torch_tensor();

    const std::string mtx_file; // matrix file: each column = data point
    const std::string idx_file; // matrix file: each column = data point
    const Index B;              // block size

    Index size() const { return B; }
    Index nfeature() const { return D; }
    Index ntot() const { return N; }

    std::tuple<Index, Index> dim() const { return { D, N }; }

private:
    Index D;                     // # features
    Index N;                     // # total samples
    std::vector<Index> idx_tab;  // matrix column index tab
    std::vector<Scalar> mem_vec; // temporary data holder
    SpMat data_DN;               // D x N sparse matrix
};

/// Populate the memory by reading the data from .mtx
/// @param subcol
void
mtx_memory_block_t::read(const std::vector<Index> &subcol)
{
    ASSERT(subcol.size() == size(), "Need the columns for " << B << " samples");

    for (Index r = 0; r < subcol.size(); ++r) {
        const Index ii = subcol.at(r);
        if (ii >= N & ii < 0)
            continue;
        for (typename SpMat::InnerIterator it(data_DN, ii); it; ++it) {
            // const Index i = it.col(); // data, sample index
            // ASSERT(i == ii, "i == ii"); // double check
            const Index k = it.row();        // data feature idx
            const Index mem_pos = k + r * D; // linearize
            mem_vec[mem_pos] = it.value();   // row-major position
        }
    }
}

/// @param mtx
/// @param idx
/// @param batch_size
mtx_memory_block_t ::mtx_memory_block_t(const std::string _mtx,
                                        const std::string _idx,
                                        const Index batch_size)
    : mtx_file(_mtx)
    , idx_file(_idx)
    , B(batch_size)
{

    ///////////////////////
    // 1. dimensionality //
    ///////////////////////

    mmutil::index::mm_info_reader_t info;
    CHK(mmutil::bgzf::peek_bgzf_header(mtx_file, info));
    D = info.rows(); // dimensionality
    N = info.cols(); // total number of samplse
    TLOG("Sparse mtx Data: " << D << " x " << N);

    //////////////////////////////////////
    // 2. Index tab for a quick look-up //
    //////////////////////////////////////
    idx_tab.clear();
    mmutil::index::read_mmutil_index(idx_file, idx_tab);
    TLOG("Read the index tab for the columns");

    // mmutil::index::check_index_tab(mtx_file, idx_tab);
    // TLOG("Checked the index tab");

    ////////////////////////////////////
    // 3. Read the full sparse matrix //
    ///////////////////////////////////

    using triplet_t = Eigen::Triplet<Scalar, Index>;
    std::vector<triplet_t> tvec;
    tvec.reserve(info.nnz());
    _triplet_reader_t<triplet_t> reader(tvec);
    visit_matrix_market_file(mtx_file, reader);
    data_DN.resize(D, N);
    data_DN.reserve(tvec.size());
    data_DN.setFromTriplets(tvec.begin(), tvec.end());
    tvec.clear();
    TLOG("Successfully loaded the sparse matrix");

    ////////////////////////////
    // 4. Pre-allocate memory //
    ////////////////////////////

    mem_vec.resize(D * B);
    std::fill(std::begin(mem_vec), std::end(mem_vec), 0.);
    TLOG("Successfully preallocated the small memory chunk");
}

/// @returns torch tensor (batch_size x feature)
torch::Tensor
mtx_memory_block_t::torch_tensor()
{
    // This must be a leaf tensor
    auto options =
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);

    return torch::from_blob(mem_vec.data(), { B, D }, options);
}

void
mtx_memory_block_t::clear()
{
    // This is faster than disk I/O
    std::fill(std::begin(mem_vec), std::end(mem_vec), 0);
}

void
create_ones_like(const mtx_memory_block_t &memory_block,
                 const std::string out_file)
{
    using T = Eigen::Triplet<mtx_memory_block_t::Scalar>;
    std::vector<T> triplets;
    const auto ntot = memory_block.ntot();
    triplets.reserve(ntot);
    for (auto j = 0; j < memory_block.ntot(); ++j) {
        triplets.emplace_back(T(0, j, 1.));
    }

    Eigen::SparseMatrix<mtx_memory_block_t::Scalar,
                        Eigen::RowMajor,
                        mtx_memory_block_t::Index>
        _covar(1, ntot);

    _covar.setFromTriplets(triplets.begin(), triplets.end());
    write_matrix_market_file(out_file, _covar);
}

} // end of namespace

#endif
