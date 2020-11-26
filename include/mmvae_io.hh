#include "mmvae.hh"

#ifndef MMVAE_IO_HH_
#define MMVAE_IO_HH_

namespace mmvae {

using Scalar = float;
using Index = std::ptrdiff_t;

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

////////////////////////////////////////////////
// A MTX file visitor that collects triplets  //
////////////////////////////////////////////////

template <typename T>
struct _triplet_reader_t {
    using scalar_t = float;
    using index_t = std::ptrdiff_t;
    using Triplet = T;
    using TripletVec = std::vector<T>;

    using index_map_t = std::unordered_map<index_t, index_t>;

    explicit _triplet_reader_t(TripletVec &_tvec,
                               index_map_t &_remap,
                               const index_t _nnz = 0)
        : Tvec(_tvec)
        , remap(_remap)
        , NNZ(_nnz)
    {
        max_row = 0;
        max_col = 0;
        max_elem = 0;
        if (NNZ > 0) {
            Tvec.reserve(NNZ);
        }
        ASSERT(remap.size() > 0, "Empty Remap");
    }

    void set_file(BGZF *_fp) { fp = _fp; }

    void eval_after_header(const index_t r, const index_t c, const index_t e)
    {
        max_row = r;
        max_col = c;
        max_elem = e;
    }

    void eval(const index_t row, const index_t col, const scalar_t weight)
    {
        if (remap.count(col) > 0) {
            Tvec.emplace_back(T(row, remap[col], weight));
        }
    }

    void eval_end_of_file()
    {
#ifdef DEBUG
        if (Tvec.size() < NNZ) {
            WLOG("This file may have lost elements : " << Tvec.size() << " vs. "
                                                       << NNZ);
        }
        TLOG("Tvec : " << Tvec.size() << " vs. " << NNZ << " vs. " << max_elem);
#endif
    }

    BGZF *fp;

    index_t max_row;
    index_t max_col;
    index_t max_elem;
    TripletVec &Tvec;
    index_map_t &remap;
    const index_t NNZ;
};

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

    std::tuple<Index, Index> dim() const { return std::tie(D, N); }

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
    using std_triplet_t = std::tuple<Index, Index, Scalar>;
    using _reader_t = _triplet_reader_t<std_triplet_t>;
    using vec_vec_t = std::vector<std::vector<Index>>;

private:
    void init();

    Index D;                    // # features
    Index N;                    // # total samples
    std::vector<Index> idx_tab; // matrix column index tab

    std::vector<Scalar> mem_vec; // temporary data holder

    _reader_t::TripletVec Tvec; // keep accumulating this
    vec_vec_t dup;              // keep track of duplicates

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

            for (auto j : dup.at(remap.at(col))) {
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

    _reader_t::index_map_t col2idx; // large index -> small index
    _reader_t::index_map_t idx2col; // vice versa

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

    // Tvec.clear();
    // for (auto block : blocks) {
    //     _reader_t reader(Tvec, col2idx);
    //     CHK(mmutil::bgzf::visit_bgzf_block(mtx_file,
    //                                        block.lb_mem,
    //                                        block.ub_mem,
    //                                        reader));
    // }

    // for (auto tt : Tvec) {
    //     Index r, k;
    //     Scalar w;
    //     std::tie(r, k, w) = tt;        //
    //     for (auto j : dup[k]) {        //
    //         const Index i = j * D + r; // column major -> row major
    //         mem_vec[i] = w;
    //     }
    // }

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

    // for (auto tt : Tvec) {
    //     Index r, k;
    //     Scalar w;
    //     std::tie(r, k, w) = tt;        //
    //     for (auto j : dup[k]) {        //
    //         const Index i = j * D + r; // column major -> row major
    //         mem_vec[i] = 0.;           // clear the info
    //     }
    // }
    // Tvec.clear();

    // for (auto block : blocks) {
    //     _mem_cleaner_t mcleaner(D, col2idx, dup, mem_vec);
    //     CHK(mmutil::bgzf::visit_bgzf_block(mtx_file,
    //                                        block.lb_mem,
    //                                        block.ub_mem,
    //                                        mcleaner));
    // }

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
    TLOG("Sparse Mtx Data: " << D << " x " << N);

    //////////////////////////////////////
    // 2. Index tab for a quick look-up //
    //////////////////////////////////////
    idx_tab.clear();
    mmutil::index::read_mmutil_index(idx_file, idx_tab);
    TLOG("Read the index tab for the columns");

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

} // namespace

#endif