#include <iostream>
#include <fstream>
#include <cstdio>

#include "io.hh"
#include "bgzf.h"
#include "kstring.h"

#ifndef MMUTIL_BGZF_UTIL_HH_
#define MMUTIL_BGZF_UTIL_HH_

namespace mmutil { namespace bgzf {

using Scalar = float;
using Index = std::ptrdiff_t;

static constexpr Index MISSING_POS = 0;
static constexpr Index LAST_POS = 0;

/// @param bgz_file : bgzipped file name
/// @param fun      : a functor with
///                   (1) set_file(FP*)
///                   (2) eval_after_header(max_row, max_col, max_nnz)
template <typename FUN>
int peek_bgzf_header(const std::string bgz_file, FUN &fun);

/// @param bgz_file : bgzipped file name
/// @param fun      : a functor with
///                   (1) set_file(FP*)
///                   (2) eval_after_header(max_row, max_col, max_nnz)
///                   (3) eval(row, col, weight)
///                   (4) eval_end_of_file()
template <typename FUN>
int visit_bgzf(const std::string bgz_file, FUN &fun);

/// @param bgz_file : bgzipped file name
/// @param beg_pos  : beginning position [beg, end)
/// @param end_pos  : ending position    [beg, end)
/// @param fun      : a functor with
///                   (1) eval(row, col, weight)
template <typename FUN>
int visit_bgzf_block(const std::string bgz_file,
                     const Index beg_pos,
                     const Index end_pos,
                     FUN &fun);

/////////////////////
// implementations //
/////////////////////

////////////////////////////////////////////////////////////////

template <typename FUN>
int
visit_bgzf_block(const std::string bgz_file,
                 const Index beg_pos,
                 const Index end_pos,
                 FUN &fun)
{

    typedef enum _state_t { S_WORD, S_EOW } state_t;
    strbuf_t strbuf;
    state_t state = S_EOW;

    Index num_cols = 0;
    Index row = 0, col = 0;
    Scalar weight = 0.;

    auto read_triplet = [&]() {
        switch (num_cols) {
        case 0:
            row = strbuf.take_uint64();
            break;
        case 1:
            col = strbuf.take_uint64();
            break;
        case 2:
            weight = strbuf.take_float();
            break;
        }
        state = S_EOW;
        strbuf.clear();
    };

    kstring_t *str = (kstring_t *)calloc(1, sizeof(kstring_t));

    BGZF *fp;

    if ((fp = bgzf_open(bgz_file.c_str(), "r")) == 0) {
        TLOG("Failed to open the file: " << bgz_file);
        return EXIT_FAILURE;
    }

    bgzf_seek(fp, beg_pos, SEEK_SET);

    ////////////////////
    // parse triplets //
    ////////////////////

    while ((bgzf_getline(fp, '\n', str)) >= 0) {

        if (str->l < 1 || str->s[0] == '%') {
            WLOG("mmutil_bgzf_util.hh: Found comment line in the middle:\n"
                 << str->s);
            state = S_EOW;
            continue; // skip comment
        }

        strbuf.clear();
        num_cols = 0;
        for (size_t pos = 0; pos < str->l; ++pos) {
            const char c = str->s[pos];
            if (std::isspace(c) && strbuf.size() > 0) {
                read_triplet();
                num_cols++;
            } else {
                strbuf.add(c);
                state = S_WORD;
            }
        }

        if (state == S_WORD) {
            read_triplet();
            num_cols++;
        }

        if (num_cols < 3) {
            WLOG("mmutil_bgzf_util.hh: Found this incomplete line: " << str->s);
            state = S_EOW;
            continue;
        }

        // one-based --> zero-based
        fun.eval(row - 1, col - 1, weight);

        // The end position may not be set
        if (end_pos != LAST_POS && bgzf_tell(fp) >= end_pos)
            break;
    }

    free(str->s);
    free(str);
    bgzf_close(fp);

    return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////

template <typename FUN>
int
peek_bgzf_header(const std::string bgz_file, FUN &fun)
{

    BGZF *fp;

    if ((fp = bgzf_open(bgz_file.c_str(), "r")) == 0) {
        TLOG("Failed to open the file: " << bgz_file);
        return EXIT_FAILURE;
    }

    fun.set_file(fp);

    kstring_t *str = (kstring_t *)calloc(1, sizeof(kstring_t));

    //////////////////////
    // parsing triplets //
    //////////////////////

    typedef enum _state_t { S_WORD, S_EOW } state_t;
    strbuf_t strbuf;
    state_t state = S_EOW;

    /////////////////////
    // read the header //
    /////////////////////

    Index max_row, max_col, max_nnz;
    int num_cols = 0;

    auto read_header_triplet = [&]() {
        switch (num_cols) {
        case 0:
            max_row = strbuf.take_uint64();
            break;
        case 1:
            max_col = strbuf.take_uint64();
            break;
        case 2:
            max_nnz = strbuf.take_uint64();
            break;
        }
        state = S_EOW;
        strbuf.clear();
    };

    int nheader = 0;

    // Read just one header line
    // Don't advance the file pointer

    while (nheader == 0 && bgzf_getline(fp, '\n', str) >= 0) {

        if (str->l < 1 || str->s[0] == '%') {
            state = S_EOW;
            continue; // skip comment
        }

        ////////////////////
        // parse triplets //
        ////////////////////

        num_cols = 0;

        for (size_t pos = 0; pos < str->l; ++pos) {
            const char c = str->s[pos];

            if (std::isspace(c) && strbuf.size() > 0) {
                read_header_triplet();
                num_cols++;
            } else {
                strbuf.add(c);
                state = S_WORD;
            }
        }

        if (state == S_WORD) {
            read_header_triplet();
            num_cols++;
        }

        if (num_cols == 3) {
            ++nheader;
        }
    }

    ASSERT(nheader == 1, "mmutil_bgzf_util.hh: Failed to read the header");

    fun.eval_after_header(max_row, max_col, max_nnz);

    free(str->s);
    free(str);
    bgzf_close(fp);

    return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////

template <typename FUN>
int
visit_bgzf(const std::string bgz_file, FUN &fun)
{

    BGZF *fp;

    if ((fp = bgzf_open(bgz_file.c_str(), "r")) == 0) {
        TLOG("Failed to open the file: " << bgz_file);
        return EXIT_FAILURE;
    }

    fun.set_file(fp);

    kstring_t *str = (kstring_t *)calloc(1, sizeof(kstring_t));

    //////////////////////
    // parsing triplets //
    //////////////////////

    typedef enum _state_t { S_WORD, S_EOW } state_t;
    strbuf_t strbuf;
    state_t state = S_EOW;

    /////////////////////
    // read the header //
    /////////////////////

    Index max_row, max_col, max_nnz;
    int num_cols = 0;

    auto read_header_triplet = [&]() {
        switch (num_cols) {
        case 0:
            max_row = strbuf.take_uint64();
            break;
        case 1:
            max_col = strbuf.take_uint64();
            break;
        case 2:
            max_nnz = strbuf.take_uint64();
            break;
        }
        state = S_EOW;
        strbuf.clear();
    };

    int nheader = 0;

    // Read just one header line
    // Don't advance the file pointer

    while (nheader == 0 && bgzf_getline(fp, '\n', str) >= 0) {

        std::cerr << str->s << std::endl;

        if (str->l < 1 || str->s[0] == '%') {
            state = S_EOW;
            continue; // skip comment
        }

        ////////////////////
        // parse triplets //
        ////////////////////

        num_cols = 0;

        for (size_t pos = 0; pos < str->l; ++pos) {
            const char c = str->s[pos];

            if (std::isspace(c) && strbuf.size() > 0) {
                read_header_triplet();
                num_cols++;
            } else {
                strbuf.add(c);
                state = S_WORD;
            }
        }

        if (state == S_WORD) {
            read_header_triplet();
            num_cols++;
        }

        if (num_cols == 3) {
            ++nheader;
        }
    }

    ASSERT(nheader == 1, "mmutil_bgzf_util.hh: Failed to read the header");

    fun.eval_after_header(max_row, max_col, max_nnz);

    ////////////////////
    // parse triplets //
    ////////////////////

    Index row = 0, col = 0;
    Scalar weight = 0.;

    auto read_triplet = [&]() {
        switch (num_cols) {
        case 0:
            row = strbuf.take_uint64();
            break;
        case 1:
            col = strbuf.take_uint64();
            break;
        case 2:
            weight = strbuf.take_float();
            break;
        }
        state = S_EOW;
        strbuf.clear();
    };

    const Index INTERVAL = 1e6;
    const Index MAX_PRINT = (max_nnz / INTERVAL);
    Index num_nz = 0;

    auto show_progress = [&num_nz, &INTERVAL, &MAX_PRINT]() {
        if (num_nz % INTERVAL == 0) {
            std::cerr << "\r" << std::left << std::setfill('.');
            std::cerr << std::setw(30) << "Reading ";
            std::cerr << std::right << std::setfill(' ') << std::setw(10)
                      << (num_nz / INTERVAL) << " x 1M triplets";
            std::cerr << " (total " << std::setw(10) << MAX_PRINT << ")";
            std::cerr << "\r" << std::flush;
        }
    };

    while (bgzf_getline(fp, '\n', str) >= 0) {

        if (str->l < 1 || str->s[0] == '%') {
            WLOG("mmutil_bgzf_util.hh: Found comment line in the middle ["
                 << num_nz << "]\n"
                 << str->s);
            state = S_EOW;
            continue; // skip comment
        }

        num_cols = 0;
        strbuf.clear();
        for (size_t pos = 0; pos < str->l; ++pos) {
            const char c = str->s[pos];
            if (std::isspace(c) && strbuf.size() > 0) {
                read_triplet();
                num_cols++;
            } else {
                strbuf.add(c);
                state = S_WORD;
            }
        }

        if (state == S_WORD) {
            read_triplet();
            num_cols++;
        }

        if (num_cols < 3) {
            WLOG(
                "mmutil_bgzf_util.hh: mmutil_bgzf_util.hh: Found this incomplete line ["
                << num_nz << "]");
            state = S_EOW;
            continue;
        }

        // one-based --> zero-based
        fun.eval(row - 1, col - 1, weight);
        ++num_nz;
        show_progress();
    }

    if (num_nz >= INTERVAL)
        std::cerr << std::endl;
    fun.eval_end_of_file();

    free(str->s);
    free(str);
    bgzf_close(fp);

    return EXIT_SUCCESS;
}

}} // namespace mmutil, bgz

#endif
