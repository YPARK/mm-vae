#include "mmvae.hh"

#ifndef MMVAE_IO_HH_
#define MMVAE_IO_HH_

namespace mmvae {

using Scalar = float;
using Index = std::ptrdiff_t;

void
write_tensor(const std::string file_, torch::Tensor param_)
{
    using Mat =
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    using Vec = Eigen::Matrix<float, Eigen::Dynamic, 1>;

    if (param_.dim() == 2) {
        Eigen::Map<Mat> param(param_.data_ptr<float>(),
                              param_.size(0),
                              param_.size(1));
        write_data_file(file_, param);
    } else if (param_.dim() < 2) {
        Eigen::Map<Vec> param(param_.data_ptr<float>(), param_.numel());
        write_data_file(file_, param);
    }
}

struct memory_block_t {
    Index lb;
    Index lb_mem;
    Index ub;
    Index ub_mem;
};

/// @param index_tab
/// @param subcol
/// @param gap
std::vector<memory_block_t>
find_consecutive_blocks(const std::vector<Index> &index_tab,
                        const std::vector<Index> &subcol,
                        const Index gap);

//////////////////////////////
// Matrix Market data block //
//////////////////////////////

struct mtx_data_block_t {

    using Index = std::ptrdiff_t;
    using Scalar = float;

    explicit mtx_data_block_t(const std::string _mtx,
                              const std::string _idx,
                              const Index batch_size)
        : mtx_file(_mtx)
        , idx_file(_idx)
        , B(batch_size)
    {
        init();
    }

    const std::string mtx_file; // matrix file: each column = data point
    const std::string idx_file; // matrix file: each column = data point

    const Index B;

    Index size() const { return B; }
    Index nfeature() const { return D; }
    Index ntot() const { return N; }

    std::tuple<Index, Index> dim() const { return { D, N }; }

    /// Populate the memory by reading the data from .mtx
    /// @param subcol
    void read(const std::vector<Index> &subcol);
    void clear();

    /// @returns torch tensor (batch_size x feature)
    torch::Tensor torch_tensor()
    {
        // This must be a leaf tensor
        auto options =
            torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);

        return torch::from_blob(mem_vec.data(), { B, D }, options);
    }

private:
    using vec_vec_t = std::vector<std::vector<Index>>;

private:
    void init();

    Index D;                     // # features
    Index N;                     // # total samples
    std::vector<Index> idx_tab;  // matrix column index tab
    std::vector<Scalar> mem_vec; // temporary data holder
    vec_vec_t dup;               // keep track of duplicates

    struct _mem_reader_t {
        using Index = std::ptrdiff_t;
        using index_map_t = std::unordered_map<Index, Index>;
        using vec_vec_t = std::vector<std::vector<Index>>;

        explicit _mem_reader_t(const Index d,
                               index_map_t &_remap,
                               vec_vec_t &_dup,
                               std::vector<Scalar> &_mem)
            : D(d)
            , remap(_remap)
            , dup(_dup)
            , mem_vec(_mem)
        {
        }

        void set_file(BGZF *_fp) { fp = _fp; }

        void eval_after_header(const Index r, const Index c, const Index e) { }

        void eval(const Index r, const Index col, const Scalar weight)
        {
            if (remap.count(col) == 0)
                return;

            for (Index j : dup.at(remap.at(col))) {
                const Index i = j * D + r; // row major position
                mem_vec[i] = weight;       // write it down
            }
        }

        void eval_end_of_file() { }

        const Index D;
        const index_map_t &remap;
        const vec_vec_t &dup;
        std::vector<Scalar> &mem_vec;
        BGZF *fp;
    };
};

////////////////////
// implementation //
////////////////////

/// @param index_tab
/// @param subcol
/// @param gap
std::vector<memory_block_t>
find_consecutive_blocks(const std::vector<Index> &index_tab,
                        const std::vector<Index> &subcol,
                        const Index gap = 10)
{

    const Index N = index_tab.size();
    ASSERT(N > 1, "Empty index map");

    std::vector<Index> sorted(subcol.size());
    std::copy(subcol.begin(), subcol.end(), sorted.begin());
    std::sort(sorted.begin(), sorted.end());

    std::vector<std::tuple<Index, Index>> intervals;
    {
        Index beg = sorted[0];
        Index end = beg;

        for (Index jj = 1; jj < sorted.size(); ++jj) {
            const Index ii = sorted[jj];
            if (ii >= (end + gap)) {                  // Is it worth adding
                intervals.emplace_back(beg, end + 1); // a new block?
                beg = ii;                             // Start a new block
                end = ii;                             // with this ii
            } else {                                  //
                end = ii;                             // Extend the current one
            }                                         // to cover this point
        }                                             //
                                                      //
        if (beg <= sorted[sorted.size() - 1]) {       // Set the upper-bound
            intervals.emplace_back(beg, end + 1);     //
        }
    }

    std::vector<memory_block_t> ret;

    for (auto intv : intervals) {

        Index lb, lb_mem, ub, ub_mem = 0;
        std::tie(lb, ub) = intv;

        if (lb >= N)
            continue;

        lb_mem = index_tab[lb];

        if (ub < N) {
            ub_mem = index_tab[ub];
        }

        ret.emplace_back(memory_block_t { lb, lb_mem, ub, ub_mem });
    }

    return ret;
}

/// Populate the memory by reading the data from .mtx
/// @param subcol
void
mtx_data_block_t::read(const std::vector<Index> &subcol)
{

    ASSERT(subcol.size() == size(), "Need the columns for " << B << " samples");

    std::vector<Index> sorted(subcol.size());
    std::copy(std::begin(subcol), std::end(subcol), std::begin(sorted));
    std::sort(std::begin(sorted), std::end(sorted));

    _mem_reader_t::index_map_t col2idx; // large index -> small index
    _mem_reader_t::index_map_t idx2col; // vice versa

    for (std::size_t j = 0; j < sorted.size(); ++j) {
        if (col2idx.count(sorted[j]) < 1) {
            std::size_t k = col2idx.size();
            col2idx[sorted[j]] = k;
            idx2col[k] = sorted[j];
        }
    }

    for (std::size_t j = 0; j < size(); ++j) {
        const auto c = subcol[j];
        const auto k = col2idx[c];
        dup[k].emplace_back(j);
    }

    const auto blocks = find_consecutive_blocks(idx_tab, subcol);

    for (auto block : blocks) {
        _mem_reader_t mreader(D, col2idx, dup, mem_vec);

        CHK(mmutil::bgzf::visit_bgzf_block(mtx_file,
                                           block.lb_mem,
                                           block.ub_mem,
                                           mreader));
    }
}

void
mtx_data_block_t::clear()
{
    // This is faster than disk I/O
    std::fill(std::begin(mem_vec), std::end(mem_vec), 0);

    for (Index k = 0; k < size(); ++k) {
        dup[k].clear();
    }
}

void
mtx_data_block_t::init()
{
    ///////////////////////
    // 1. dimensionality //
    ///////////////////////
    mmutil::index::mm_info_reader_t info;
    CHK(mmutil::bgzf::peek_bgzf_header(mtx_file, info));
    D = info.rows(); // dimensionality
    N = info.cols(); // total number of samplse
    TLOG("Sparse Mtx Data: " << D << " x " << N << " from " << mtx_file);

    //////////////////////////////////////
    // 2. Index tab for a quick look-up //
    //////////////////////////////////////
    idx_tab.clear();
    mmutil::index::read_mmutil_index(idx_file, idx_tab);
    TLOG("Read the index tab for the columns: " << idx_file);

    // mmutil::index::check_index_tab(mtx_file, idx_tab);
    // TLOG("Checked the index tab");

    ////////////////////////////
    // 3. Pre-allocate memory //
    ////////////////////////////
    mem_vec.resize(D * B);
    std::fill(std::begin(mem_vec), std::end(mem_vec), 0.);

    dup.reserve(size());
    for (Index k = 0; k < size(); ++k) {
        dup.emplace_back(std::vector<Index> {});
    }
}

void
create_ones_like(const mtx_data_block_t &data_block, const std::string out_file)
{
    using T = Eigen::Triplet<mtx_data_block_t::Scalar>;
    std::vector<T> triplets;
    const auto ntot = data_block.ntot();
    triplets.reserve(ntot);
    for (auto j = 0; j < data_block.ntot(); ++j) {
        triplets.emplace_back(T(0, j, 1.));
    }

    Eigen::SparseMatrix<mtx_data_block_t::Scalar,
                        Eigen::RowMajor,
                        mtx_data_block_t::Index>
        _covar(1, ntot);

    _covar.setFromTriplets(triplets.begin(), triplets.end());
    write_matrix_market_file(out_file, _covar);
}

} // namespace

#endif
