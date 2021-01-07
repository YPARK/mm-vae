#include <string>
#include <unordered_map>
#include "bgzf.h"

#ifndef IO_VISITOR_HH_
#define IO_VISITOR_HH_

////////////////////////////////////////////////
// A MTX file visitor that collects triplets  //
////////////////////////////////////////////////

template <typename T>
struct _triplet_reader_t {
    using scalar_t = float;
    using index_t = std::ptrdiff_t;
    using Triplet = T;
    using TripletVec = std::vector<T>;

    explicit _triplet_reader_t(TripletVec &_tvec)
        : Tvec(_tvec)
    {
        max_row = 0;
        max_col = 0;
        max_elem = 0;
        TLOG("Start reading a list of triplets");
    }

    void set_fp(BGZF *_fp) { fp = _fp; }

    void eval_after_header(const index_t r, const index_t c, const index_t e)
    {
        max_row = r;
        max_col = c;
        max_elem = e;
        Tvec.reserve(max_elem);
    }

    void eval(const index_t row, const index_t col, const scalar_t weight)
    {
        Tvec.emplace_back(T(row, col, weight));
    }

    void eval_end_of_file()
    {
        if (Tvec.size() < max_elem) {
            WLOG("This file may have lost elements : " << Tvec.size() << " vs. "
                                                       << max_elem);
        }
        TLOG("Finished reading a list of triplets");
    }

    BGZF *fp;

    index_t max_row;
    index_t max_col;
    index_t max_elem;
    TripletVec &Tvec;
};

////////////////////////////////////////////////
// A MTX file visitor that collects triplets  //
////////////////////////////////////////////////

template <typename T>
struct _triplet_reader_remap_t {
    using scalar_t = float;
    using index_t = std::ptrdiff_t;
    using Triplet = T;
    using TripletVec = std::vector<T>;

    using index_map_t = std::unordered_map<index_t, index_t>;

    explicit _triplet_reader_remap_t(TripletVec &_tvec,
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

#endif
