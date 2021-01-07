#include <string>

#include "gzstream.hh"
#include "bgzstream.hh"
#include "strbuf.hh"
#include "util.hh"
#include "bgzf.h"

#ifndef IO_ALG_HH_
#define IO_ALG_HH_

template <typename FUN>
void visit_matrix_market_file(const std::string filename, FUN &fun);

template <typename IFS, typename FUN>
void visit_matrix_market_stream(IFS &ifs, FUN &fun);

/////////////////////
// implementations //
/////////////////////

template <typename IFS, typename FUN>
void
visit_matrix_market_stream(IFS &ifs, FUN &fun)
{
    using scalar_t = typename FUN::scalar_t;
    using index_t = typename FUN::index_t;

    //////////////////////////
    // Finite state machine //
    //////////////////////////

    typedef enum _state_t { S_COMMENT, S_WORD, S_EOW, S_EOL } state_t;
    const char eol = '\n';
    const char comment = '%';

    /////////////////////////
    // use buffered stream //
    /////////////////////////

    std::istreambuf_iterator<char> END;
    std::istreambuf_iterator<char> it(ifs);

    strbuf_t strbuf;
    state_t state = S_EOL;

    size_t num_nz = 0u;   // number of non-zero elements
    size_t num_rows = 0u; // number of rows
    size_t num_cols = 0u; // number of columns

    /////////////////////////////////////
    // read the first line and headers //
    /////////////////////////////////////

    // %%MatrixMarket matrix coordinate integer general

    std::vector<index_t> dims(3);

    auto extract_idx_word = [&]() {
        const index_t _idx = strbuf.get<index_t>();
        if (num_cols < dims.size()) {
            dims[num_cols] = _idx;
        }
        state = S_EOW;
        strbuf.clear();
        return _idx;
    };

    for (; num_rows < 1 && it != END; ++it) {
        char c = *it;

        // Skip the comment line. It doesn't count toward the line
        // count, and we don't bother reading the content.
        if (state == S_COMMENT) {
            if (c == eol) {
                state = S_EOL;
#ifdef DEBUG
                std::cerr << std::endl;
#endif
            }
#ifdef DEBUG
            std::cerr << c;
#endif
            continue;
        }

        if (c == comment) {
            state = S_COMMENT;
#ifdef DEBUG
            std::cerr << c;
#endif
            continue;
        }

        // Do the regular parsing of a triplet

        if (c == eol) {
            if (state == S_WORD) {
                std::ignore = extract_idx_word();
                num_cols++;
            }

            state = S_EOL;
            num_rows++;

        } else if (isspace(c)) {
            std::ignore = extract_idx_word();
            num_cols++;
        } else {
            strbuf.add(c);
            state = S_WORD;
        }
    }

    fun.eval_after_header(dims[0], dims[1], dims[2]);

    /////////////////////////////
    // Read a list of triplets //
    /////////////////////////////

    index_t row = 0, col = 0;
    scalar_t weight = 0.;

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

    num_cols = 0;
    num_nz = 0;

    const index_t max_row = dims[0];
    const index_t max_col = dims[1];
    const index_t max_elem = dims[2];
    const index_t INTERVAL = 1e6;
    const index_t MAX_PRINT = (max_elem / INTERVAL);

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

    for (; num_nz < static_cast<size_t>(max_elem) && it != END; ++it) {
        char c = *it;

        // Skip the comment line. It doesn't count toward the line
        // count, and we don't bother reading the content.
        if (state == S_COMMENT) {
            if (c == eol)
                state = S_EOL;
            continue;
        }

        if (c == comment) {
            state = S_COMMENT;
            WLOG("Found a commented line in the middle");
            continue;
        }

        // Do the regular parsing of a triplet

        if (c == eol) {
            if (state == S_WORD) {
                read_triplet();
                num_cols++;
            }

            state = S_EOL;

            if (row <= 0 || row > max_row) {
                WLOG("Ignore row" << std::setfill(' ') << std::setw(10) << row);
                continue;
            } else if (col <= 0 || col > max_col) {
                WLOG("Ignore column" << std::setfill(' ') << std::setw(10)
                                     << col);
                continue;
            } else {
                // convert 1-based to 0-based
                fun.eval(row - 1, col - 1, weight);
                ++num_nz;
            }

            show_progress();
            num_cols = 0;

        } else if (isspace(c) && strbuf.size() > 0) {
            read_triplet();
            num_cols++;
        } else {
            strbuf.add(c);
            state = S_WORD;
        }
    }
    if (num_nz >= INTERVAL)
        std::cerr << std::endl;
    fun.eval_end_of_file();
} // End of the visit

template <typename FUN>
void
visit_matrix_market_file(const std::string filename, FUN &fun)
{
    if (bgzf_is_bgzf(filename.c_str()) > 0) {

        ibgzf_stream ifs(filename.c_str(), std::ios::in);
        visit_matrix_market_stream(ifs, fun);
        ifs.close();

    } else if (filename.size() >= 3 &&
               (filename.substr(filename.size() - 3) == ".gz")) {
        igzstream ifs(filename.c_str(), std::ios::in);
        visit_matrix_market_stream(ifs, fun);
        ifs.close();
    } else {
        std::ifstream ifs(filename.c_str(), std::ios::in);
        visit_matrix_market_stream(ifs, fun);
        ifs.close();
    }
}

#endif
