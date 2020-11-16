////////////////////////////////////////////////////////////////
// I/O routines
#include <cctype>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Sparse>
// #include <filesystem> c++17
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "io_visitor.hh"
#include "eigen_util.hh"
#include "gzstream.hh"
#include "bgzstream.hh"
#include "strbuf.hh"
#include "tuple_util.hh"
#include "util.hh"
#include "bgzf.h"

#ifndef UTIL_IO_HH_
#define UTIL_IO_HH_

bool
file_exists(std::string filename)
{
    std::ifstream f(filename.c_str());
    return f.good();
}

bool
all_files_exist(std::vector<std::string> filenames)
{
    bool ret = true;
    for (auto f : filenames) {
        if (!file_exists(f)) {
            TLOG(std::left << std::setw(10) << "Missing: " << std::setw(30)
                           << f);
            ret = false;
        }
        TLOG(std::left << std::setw(10) << "Found: " << std::setw(30) << f);
    }
    return ret;
}

/////////////////////////////////
// common utility for data I/O //
/////////////////////////////////

bool
is_file_gz(const std::string filename)
{
    if (filename.size() < 3)
        return false;
    return filename.substr(filename.size() - 3) == ".gz";
}

bool
is_file_bgz(const std::string filename)
{
    if (bgzf_is_bgzf(filename.c_str()) < 1)
        return false;
    return true;
}

std::shared_ptr<std::ifstream>
open_ifstream(const std::string filename)
{
    std::shared_ptr<std::ifstream> ret(
        new std::ifstream(filename.c_str(), std::ios::in));
    return ret;
}

std::shared_ptr<igzstream>
open_igzstream(const std::string filename)
{
    std::shared_ptr<igzstream> ret(
        new igzstream(filename.c_str(), std::ios::in));
    return ret;
}

///////////////
// I/O pairs //
///////////////

template <typename IFS, typename T1, typename T2>
auto
read_dict_stream(IFS &ifs, std::unordered_map<T1, T2> &in)
{
    in.clear();
    T1 v;
    T2 w;
    while (ifs >> v >> w) {
        in[v] = w;
    }
    ERR_RET(in.size() == 0, "empty file");
    return EXIT_SUCCESS;
}

template <typename T1, typename T2>
auto
read_dict_file(const std::string filename, std::unordered_map<T1, T2> &in)
{
    auto ret = EXIT_SUCCESS;

    if (is_file_gz(filename)) {
        igzstream ifs(filename.c_str(), std::ios::in);
        ret = read_dict_stream(ifs, in);
        ifs.close();
    } else {
        std::ifstream ifs(filename.c_str(), std::ios::in);
        ret = read_dict_stream(ifs, in);
        ifs.close();
    }
    return ret;
}

template <typename IFS, typename T1, typename T2>
auto
read_pair_stream(IFS &ifs, std::vector<std::tuple<T1, T2>> &in)
{
    in.clear();
    T1 v;
    T2 w;
    while (ifs >> v >> w) {
        in.emplace_back(std::make_tuple(v, w));
    }
    ERR_RET(in.size() == 0, "empty file");
    return EXIT_SUCCESS;
}

template <typename T1, typename T2>
auto
read_pair_file(const std::string filename, std::vector<std::tuple<T1, T2>> &in)
{
    auto ret = EXIT_SUCCESS;

    if (is_file_gz(filename)) {
        igzstream ifs(filename.c_str(), std::ios::in);
        ret = read_pair_stream(ifs, in);
        ifs.close();
    } else {
        std::ifstream ifs(filename.c_str(), std::ios::in);
        ret = read_pair_stream(ifs, in);
        ifs.close();
    }
    return ret;
}

template <typename IFS, typename T>
auto
read_vector_stream(IFS &ifs, std::vector<T> &in)
{
    in.clear();
    T v;
    while (ifs >> v) {
        in.push_back(v);
    }
    ERR_RET(in.size() == 0, "empty vector");
    return EXIT_SUCCESS;
}

template <typename T>
auto
read_vector_file(const std::string filename, std::vector<T> &in)
{
    auto ret = EXIT_SUCCESS;

    if (is_file_gz(filename)) {
        igzstream ifs(filename.c_str(), std::ios::in);
        ret = read_vector_stream(ifs, in);
        ifs.close();
    } else {
        std::ifstream ifs(filename.c_str(), std::ios::in);
        ret = read_vector_stream(ifs, in);
        ifs.close();
    }
    return ret;
}

///////////////////
// simple writer //
///////////////////

template <typename OFS, typename Derived>
void
write_matrix_market_stream(OFS &ofs,
                           const Eigen::SparseMatrixBase<Derived> &out)
{

    const Derived &M = out.derived();

    ofs << "%%MatrixMarket matrix coordinate integer general" << std::endl;
    ofs << M.rows() << " " << M.cols() << " " << M.nonZeros() << std::endl;

    using Index = typename Derived::Index;
    using Scalar = typename Derived::Scalar;
    using SpMat = Eigen::SparseMatrix<Scalar, Eigen::RowMajor, std::ptrdiff_t>;

    // Should this be sorted by the column (the 2nd element)
    if (M.IsRowMajor) {
        SpMat Mt = M.transpose();

        for (auto k = 0; k < Mt.outerSize(); ++k) {
            for (typename Derived::InnerIterator it(Mt, k); it; ++it) {
                const Index i = it.row() + 1; // fix zero-based to one-based
                const Index j = it.col() + 1; // fix zero-based to one-based
                const auto v = it.value();
                ofs << j << " " << i << " " << v << std::endl;
            }
        }
    } else {
        for (auto k = 0; k < M.outerSize(); ++k) {
            for (typename Derived::InnerIterator it(M, k); it; ++it) {
                const Index i = it.row() + 1; // fix zero-based to one-based
                const Index j = it.col() + 1; // fix zero-based to one-based
                const auto v = it.value();
                ofs << i << " " << j << " " << v << std::endl;
            }
        }
    }
}

template <typename Derived>
void
write_matrix_market_file(const std::string filename,
                         const Eigen::SparseMatrixBase<Derived> &out)
{
    if (is_file_gz(filename)) {
        obgzf_stream ofs(filename.c_str(), std::ios::out);
        write_matrix_market_stream(ofs, out);
        ofs.close();
    } else {
        std::ofstream ofs(filename.c_str(), std::ios::out);
        write_matrix_market_stream(ofs, out);
        ofs.close();
    }
}

/////////////////////////////////////
// frequently used output routines //
/////////////////////////////////////

template <typename OFS, typename Vec>
void
write_tuple_stream(OFS &ofs, const Vec &vec)
{
    int i = 0;

    auto _print = [&ofs, &i](const auto x) {
        if (i > 0)
            ofs << " ";
        ofs << x;
        i++;
    };

    for (auto pp : vec) {
        i = 0;
        func_apply(_print, std::move(pp));
        ofs << std::endl;
    }
}

template <typename Vec>
void
write_tuple_file(const std::string filename, const Vec &out)
{
    if (is_file_gz(filename)) {
        ogzstream ofs(filename.c_str(), std::ios::out);
        write_tuple_stream(ofs, out);
        ofs.close();
    } else {
        std::ofstream ofs(filename.c_str(), std::ios::out);
        write_tuple_stream(ofs, out);
        ofs.close();
    }
}

template <typename OFS, typename Vec>
void
write_pair_stream(OFS &ofs, const Vec &vec)
{

    for (auto pp : vec) {
        ofs << std::get<0>(pp) << " " << std::get<1>(pp) << std::endl;
    }
}

template <typename Vec>
void
write_pair_file(const std::string filename, const Vec &out)
{
    if (is_file_gz(filename)) {
        ogzstream ofs(filename.c_str(), std::ios::out);
        write_pair_stream(ofs, out);
        ofs.close();
    } else {
        std::ofstream ofs(filename.c_str(), std::ios::out);
        write_pair_stream(ofs, out);
        ofs.close();
    }
}

template <typename OFS, typename Vec>
void
write_vector_stream(OFS &ofs, const Vec &vec)
{

    for (auto pp : vec) {
        ofs << pp << std::endl;
    }
}

template <typename Vec>
void
write_vector_file(const std::string filename, const Vec &out)
{
    if (is_file_gz(filename)) {
        ogzstream ofs(filename.c_str(), std::ios::out);
        write_vector_stream(ofs, out);
        ofs.close();
    } else {
        std::ofstream ofs(filename.c_str(), std::ios::out);
        write_vector_stream(ofs, out);
        ofs.close();
    }
}

/////////////////////////////
// identify dimensionality //
/////////////////////////////

template <typename IFS>
auto
num_cols(IFS &ifs)
{
    std::istreambuf_iterator<char> eos;
    std::istreambuf_iterator<char> it(ifs);
    const char eol = '\n';

    std::size_t ret = 1;
    for (; it != eos && *it != eol; ++it) {
        char c = *it;
        if (isspace(c) && c != eol)
            ++ret;
    }

    return ret;
}

template <typename IFS>
auto
num_rows(IFS &ifs)
{
    std::istreambuf_iterator<char> eos;
    std::istreambuf_iterator<char> it(ifs);
    const char eol = '\n';

    std::size_t ret = 0;
    for (; it != eos; ++it)
        if (*it == eol)
            ++ret;

    return ret;
}

template <typename IFS>
auto
num_rows_cols(IFS &ifs)
{
    std::istreambuf_iterator<char> eos;
    std::istreambuf_iterator<char> it(ifs);
    const char eol = '\n';

    std::size_t nr = 0;
    std::size_t nc = 1;
    bool start_newline = true;

    for (; it != eos; ++it) {
        std::size_t _nc = 1;
        char c = *it;

        if (*it == eol) {
            ++nr;
            nc = std::max(nc, _nc);
            start_newline = true;
        } else {
            start_newline = false;
        }

        if (isspace(c) && c != eol)
            _nc++;
    }

    return std::make_tuple(nr, nc);
}

template <typename IFS, typename T>
auto
read_data_stream(IFS &ifs, T &in)
{
    typedef typename T::Scalar elem_t;

    typedef enum _state_t { S_WORD, S_EOW, S_EOL } state_t;
    const auto eol = '\n';
    std::istreambuf_iterator<char> END;
    std::istreambuf_iterator<char> it(ifs);

    std::vector<elem_t> data;
    strbuf_t strbuf;
    state_t state = S_EOL;

    auto nr = 0u; // number of rows
    auto nc = 1u; // number of columns

    elem_t val;
    auto nmissing = 0;

    for (; it != END; ++it) {
        char c = *it;

        if (c == eol) {
            if (state == S_WORD) {
                val = strbuf.lexical_cast<elem_t>();

                if (!std::isfinite(val))
                    nmissing++;

                data.push_back(val);
                strbuf.clear();
            } else if (state == S_EOW) {
                data.push_back(NAN);
                nmissing++;
            }
            state = S_EOL;
            nr++;
        } else if (isspace(c)) {
            if (state == S_WORD) {
                val = strbuf.lexical_cast<elem_t>();

                if (!std::isfinite(val))
                    nmissing++;

                data.push_back(val);
                strbuf.clear();
            } else {
                data.push_back(NAN);
                nmissing++;
            }
            state = S_EOW;
            if (nr == 0)
                nc++;

        } else {
            strbuf.add(c);
            state = S_WORD;
        }
    }

#ifdef DEBUG
    TLOG("Found " << nmissing << " missing values");
#endif

    auto mtot = data.size();
    ERR_RET(mtot != (nr * nc),
            "# data points: " << mtot << " elements in " << nr << " x " << nc
                              << " matrix");
    ERR_RET(mtot < 1, "empty file");
    ERR_RET(nr < 1, "zero number of rows; incomplete line?");
    in = Eigen::Map<T>(data.data(), nc, nr);
    in.transposeInPlace();

    return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////
template <typename T>
auto
read_data_file(const std::string filename, T &in)
{
    auto ret = EXIT_SUCCESS;

    if (is_file_gz(filename)) {
        igzstream ifs(filename.c_str(), std::ios::in);
        ret = read_data_stream(ifs, in);
        ifs.close();
    } else {
        std::ifstream ifs(filename.c_str(), std::ios::in);
        ret = read_data_stream(ifs, in);
        ifs.close();
    }

    return ret;
}

////////////////////////////////////////////////////////////////
template <typename T>
auto
read_data_file(const std::string filename)
{
    typename std::shared_ptr<T> ret(new T {});
    auto &in = *ret.get();

    if (is_file_gz(filename)) {
        igzstream ifs(filename.c_str(), std::ios::in);
        CHK_ERR_EXIT(read_data_stream(ifs, in), "Failed to read " << filename);
        ifs.close();
    } else {
        std::ifstream ifs(filename.c_str(), std::ios::in);
        CHK_ERR_EXIT(read_data_stream(ifs, in), "Failed to read " << filename);
        ifs.close();
    }

    return ret;
}

////////////////////////////////////////////////////////////////
template <typename OFS, typename Derived>
void
write_data_stream(OFS &ofs, const Eigen::MatrixBase<Derived> &out)
{

    const Derived &M = out.derived();
    using Index = typename Derived::Index;

    for (Index r = 0u; r < M.rows(); ++r) {
        ofs << M.coeff(r, 0);
        for (Index c = 1u; c < M.cols(); ++c)
            ofs << " " << M.coeff(r, c);
        ofs << std::endl;
    }
}

template <typename OFS, typename Derived>
void
write_data_stream(OFS &ofs, const Eigen::SparseMatrixBase<Derived> &out)
{

    const Derived &M = out.derived();
    using Index = typename Derived::Index;
    using Scalar = typename Derived::Scalar;

    // Not necessarily column major
    for (Index k = 0; k < M.outerSize(); ++k) {
        for (typename Derived::InnerIterator it(M, k); it; ++it) {
            const Index i = it.row();
            const Index j = it.col();
            const Scalar v = it.value();
            ofs << i << " " << j << " " << v << std::endl;
        }
    }
}

////////////////////////////////////////////////////////////////
template <typename T>
void
write_data_file(const std::string filename, const T &out)
{
    if (is_file_gz(filename)) {
        ogzstream ofs(filename.c_str(), std::ios::out);
        write_data_stream(ofs, out);
        ofs.close();
    } else {
        std::ofstream ofs(filename.c_str(), std::ios::out);
        write_data_stream(ofs, out);
        ofs.close();
    }
}

#endif
