#include <algorithm>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Sparse>
#include <functional>
#include <vector>

#include "utils/math.hh"
#include "utils/util.hh"

#ifndef EIGEN_UTIL_HH_
#define EIGEN_UTIL_HH_

template <typename EigenVec>
inline auto
std_vector(const EigenVec eigen_vec)
{
    std::vector<typename EigenVec::Scalar> ret(eigen_vec.size());
    for (typename EigenVec::Index j = 0; j < eigen_vec.size(); ++j) {
        ret[j] = eigen_vec(j);
    }
    return ret;
}

template <typename T>
inline auto
eigen_vector(const std::vector<T> &std_vec)
{
    Eigen::Matrix<T, Eigen::Dynamic, 1> ret(std_vec.size());

    for (std::size_t j = 0; j < std_vec.size(); ++j) {
        ret(j) = std_vec.at(j);
    }

    return ret;
}

template <typename T, typename T2>
inline auto
eigen_vector_(const std::vector<T> &std_vec)
{
    Eigen::Matrix<T2, Eigen::Dynamic, 1> ret(std_vec.size());

    for (std::size_t j = 0; j < std_vec.size(); ++j) {
        ret(j) = std_vec.at(j);
    }

    return ret;
}

template <typename EigenVec, typename StdVec>
inline void
std_vector(const EigenVec eigen_vec, StdVec &ret)
{
    ret.resize(eigen_vec.size());
    using T = typename StdVec::value_type;
    for (typename EigenVec::Index j = 0; j < eigen_vec.size(); ++j) {
        ret[j] = static_cast<T>(eigen_vec(j));
    }
}

template <typename T>
inline std::vector<Eigen::Triplet<float>>
eigen_triplets(const std::vector<T> &Tvec, bool weighted = true)
{
    using Scalar = float;
    using _Triplet = Eigen::Triplet<Scalar>;
    using _TripletVec = std::vector<_Triplet>;

    _TripletVec ret;
    ret.reserve(Tvec.size());

    if (weighted) {
        for (auto tt : Tvec) {
            ret.emplace_back(
                _Triplet(std::get<0>(tt), std::get<1>(tt), std::get<2>(tt)));
        }
    } else {
        for (auto tt : Tvec) {
            ret.emplace_back(_Triplet(std::get<0>(tt), std::get<1>(tt), 1.0));
        }
    }
    return ret;
}

template <typename Scalar>
inline auto
eigen_triplets(const std::vector<Eigen::Triplet<Scalar>> &Tvec)
{
    return Tvec;
}

template <typename TVEC, typename INDEX>
inline Eigen::SparseMatrix<float, Eigen::RowMajor, std::ptrdiff_t> //
build_eigen_sparse(const TVEC &Tvec, const INDEX max_row, const INDEX max_col)
{
    auto _tvec = eigen_triplets(Tvec);
    using Scalar = float;
    using SpMat = Eigen::SparseMatrix<Scalar, Eigen::RowMajor, std::ptrdiff_t>;

    SpMat ret(max_row, max_col);
    ret.reserve(_tvec.size());
    ret.setFromTriplets(_tvec.begin(), _tvec.end());
    return ret;
}

template <typename Vec>
inline std::vector<typename Vec::Index>
eigen_argsort_descending(const Vec &data)
{
    using Index = typename Vec::Index;
    std::vector<Index> index(data.size());
    std::iota(std::begin(index), std::end(index), 0);
    std::sort(std::begin(index), std::end(index), [&](Index lhs, Index rhs) {
        return data(lhs) > data(rhs);
    });
    return index;
}

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar,
                     Eigen::Dynamic,
                     Eigen::Dynamic,
                     Eigen::ColMajor>
row_score_degree(const Eigen::SparseMatrixBase<Derived> &_xx)
{
    const Derived &xx = _xx.derived();
    using Scalar = typename Derived::Scalar;
    using Mat =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    return xx.unaryExpr([](const Scalar x) { return std::abs(x); }) *
        Mat::Ones(xx.cols(), 1);
}

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar,
                     Eigen::Dynamic,
                     Eigen::Dynamic,
                     Eigen::ColMajor>
row_score_sd(const Eigen::SparseMatrixBase<Derived> &_xx)
{
    const Derived &xx = _xx.derived();
    using Scalar = typename Derived::Scalar;
    using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Mat =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    Vec s1 = xx * Mat::Ones(xx.cols(), 1);
    Vec s2 = xx.cwiseProduct(xx) * Mat::Ones(xx.cols(), 1);
    const Scalar n = xx.cols();
    Vec ret = s2 - s1.cwiseProduct(s1 / n);
    ret = ret / std::max(n - 1.0, 1.0);
    ret = ret.cwiseSqrt();

    return ret;
}

template <typename Derived, typename ROWS>
inline Eigen::SparseMatrix<typename Derived::Scalar, //
                           Eigen::RowMajor,          //
                           std::ptrdiff_t>
row_sub(const Eigen::SparseMatrixBase<Derived> &_mat, const ROWS &sub_rows)
{
    using SpMat = typename Eigen::
        SparseMatrix<typename Derived::Scalar, Eigen::RowMajor, std::ptrdiff_t>;

    using Index = typename SpMat::Index;
    using Scalar = typename SpMat::Scalar;
    const SpMat &mat = _mat.derived();

    SpMat ret(sub_rows.size(), mat.cols());

    using ET = Eigen::Triplet<Scalar>;

    std::vector<ET> triples;

    Index rr = 0;

    for (Index r : sub_rows) {
        if (r < 0 || r >= mat.rows())
            continue;

        for (typename SpMat::InnerIterator it(mat, r); it; ++it) {
            triples.push_back(ET(rr, it.col(), it.value()));
        }

        ++rr;
    }

    ret.reserve(triples.size());
    ret.setFromTriplets(triples.begin(), triples.end());

    return ret;
}

template <typename Derived, typename ROWS>
inline Eigen::Matrix<typename Derived::Scalar, //
                     Eigen::Dynamic,           //
                     Eigen::Dynamic,           //
                     Eigen::ColMajor>
row_sub(const Eigen::MatrixBase<Derived> &_mat, const ROWS &sub_rows)
{
    using Mat = typename Eigen::Matrix<typename Derived::Scalar, //
                                       Eigen::Dynamic,           //
                                       Eigen::Dynamic,           //
                                       Eigen::ColMajor>;

    using Index = typename Mat::Index;
    using Scalar = typename Mat::Scalar;
    const Mat &mat = _mat.derived();

    Mat ret(sub_rows.size(), mat.cols());
    ret.setZero();

    Index rr = 0;

    for (Index r : sub_rows) {
        if (r < 0 || r >= mat.rows())
            continue;

        ret.row(rr) += mat.row(r);

        ++rr;
    }

    return ret;
}

template <typename Derived>
inline Eigen::
    SparseMatrix<typename Derived::Scalar, Eigen::RowMajor, std::ptrdiff_t>
    vcat(const Eigen::SparseMatrixBase<Derived> &_upper,
         const Eigen::SparseMatrixBase<Derived> &_lower)
{
    using Scalar = typename Derived::Scalar;
    using Index = typename Derived::Index;
    const Derived &upper = _upper.derived();
    const Derived &lower = _lower.derived();

    ASSERT(upper.cols() == lower.cols(), "mismatching columns in vcat");

    using _Triplet = Eigen::Triplet<Scalar>;

    std::vector<_Triplet> triplets;
    triplets.reserve(upper.nonZeros() + lower.nonZeros());

    using SpMat = typename Eigen::
        SparseMatrix<typename Derived::Scalar, Eigen::RowMajor, std::ptrdiff_t>;

    for (Index k = 0; k < upper.outerSize(); ++k) {
        for (typename SpMat::InnerIterator it(upper, k); it; ++it) {
            triplets.emplace_back(it.row(), it.col(), it.value());
        }
    }

    for (Index k = 0; k < lower.outerSize(); ++k) {
        for (typename SpMat::InnerIterator it(lower, k); it; ++it) {
            triplets.emplace_back(upper.rows() + it.row(),
                                  it.col(),
                                  it.value());
        }
    }

    SpMat result(lower.rows() + upper.rows(), upper.cols());
    result.setFromTriplets(triplets.begin(), triplets.end());
    return result;
}

template <typename Derived>
inline Eigen::
    SparseMatrix<typename Derived::Scalar, Eigen::RowMajor, std::ptrdiff_t>
    hcat(const Eigen::SparseMatrixBase<Derived> &_left,
         const Eigen::SparseMatrixBase<Derived> &_right)
{
    using Scalar = typename Derived::Scalar;
    using Index = typename Derived::Index;
    const Derived &left = _left.derived();
    const Derived &right = _right.derived();

    ASSERT(left.rows() == right.rows(), "mismatching rows in hcat");

    using _Triplet = Eigen::Triplet<Scalar>;

    std::vector<_Triplet> triplets;
    triplets.reserve(left.nonZeros() + right.nonZeros());

    using SpMat = typename Eigen::
        SparseMatrix<typename Derived::Scalar, Eigen::RowMajor, std::ptrdiff_t>;

    for (Index k = 0; k < left.outerSize(); ++k) {
        for (typename SpMat::InnerIterator it(left, k); it; ++it) {
            triplets.emplace_back(it.row(), it.col(), it.value());
        }
    }

    for (Index k = 0; k < right.outerSize(); ++k) {
        for (typename SpMat::InnerIterator it(right, k); it; ++it) {
            triplets.emplace_back(it.row(), left.cols() + it.col(), it.value());
        }
    }

    SpMat result(left.rows(), left.cols() + right.cols());
    result.setFromTriplets(triplets.begin(), triplets.end());
    return result;
}

template <typename Derived, typename Derived2>
Eigen::Matrix<typename Derived::Scalar,
              Eigen::Dynamic,
              Eigen::Dynamic,
              Eigen::ColMajor>
hcat(const Eigen::MatrixBase<Derived> &_left,
     const Eigen::MatrixBase<Derived2> &_right)
{

    using Scalar = typename Derived::Scalar;
    using Index = typename Derived::Index;

    using Mat =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    const Derived &L = _left.derived();
    const Derived2 &R = _right.derived();

    ASSERT(L.rows() == R.rows(), "Must have the same number of rows");

    Mat ret(L.rows(), L.cols() + R.cols());

    for (Index j = 0; j < L.cols(); ++j) {
        ret.col(j) = L.col(j);
    }

    for (Index j = 0; j < R.cols(); ++j) {
        ret.col(j + L.cols()) = R.col(j);
    }

    return ret;
}

template <typename Derived>
inline typename Derived::Scalar
log_sum_exp(const Eigen::MatrixBase<Derived> &log_vec)
{
    using Scalar = typename Derived::Scalar;
    using Index = typename Derived::Index;

    const Derived &xx = log_vec.derived();

    Scalar maxlogval = xx(0);
    for (Index j = 1; j < xx.size(); ++j) {
        if (xx(j) > maxlogval)
            maxlogval = xx(j);
    }

    Scalar ret = 0;
    for (Index j = 0; j < xx.size(); ++j) {
        ret += fasterexp(xx(j) - maxlogval);
    }
    ret = fasterlog(ret) + maxlogval;
    return ret;
}

template <typename Derived, typename Derived2>
inline typename Derived::Scalar
normalized_exp(const Eigen::MatrixBase<Derived> &_log_vec,
               Eigen::MatrixBase<Derived2> &_ret)
{
    using Scalar = typename Derived::Scalar;
    using Index = typename Derived::Index;

    const Derived &log_vec = _log_vec.derived();
    Derived &ret = _ret.derived();

    Index argmax;
    const Scalar log_denom = log_vec.maxCoeff(&argmax);

    auto _exp = [&log_denom](const Scalar log_z) {
        return fasterexp(log_z - log_denom);
    };

    ret = log_vec.unaryExpr(_exp).eval();
    const Scalar denom = ret.sum();
    ret /= denom;

    const Scalar log_normalizer = fasterlog(denom) + log_denom;
    return log_normalizer;
}

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar,
                     Eigen::Dynamic,
                     Eigen::Dynamic,
                     Eigen::ColMajor>
standardize(const Eigen::MatrixBase<Derived> &Xraw,
            const typename Derived::Scalar EPS = 1e-8)
{
    using Index = typename Derived::Index;
    using Scalar = typename Derived::Scalar;
    using mat_t =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using RowVec = typename Eigen::internal::plain_row_type<Derived>::type;

    mat_t X(Xraw.rows(), Xraw.cols());

    ////////////////
    // Remove NaN //
    ////////////////

    const RowVec num_obs = Xraw.unaryExpr([](const Scalar x) {
                                   return std::isfinite(x) ?
                                       static_cast<Scalar>(1) :
                                       static_cast<Scalar>(0);
                               })
                               .colwise()
                               .sum();

    X = Xraw.unaryExpr([](const Scalar x) {
        return std::isfinite(x) ? x : static_cast<Scalar>(0);
    });

    //////////////////////////
    // calculate statistics //
    //////////////////////////

    const RowVec x_mean = X.colwise().sum().cwiseQuotient(num_obs);
    const RowVec x2_mean =
        X.cwiseProduct(X).colwise().sum().cwiseQuotient(num_obs);
    const RowVec x_sd = (x2_mean - x_mean.cwiseProduct(x_mean)).cwiseSqrt();

    // standardize
    for (Index j = 0; j < X.cols(); ++j) {
        const Scalar mu = x_mean(j);
        const Scalar sd = x_sd(j) + EPS;
        auto std_op = [&mu, &sd](const Scalar &x) -> Scalar {
            const Scalar ret = static_cast<Scalar>((x - mu) / sd);
            return ret;
        };

        // This must be done with original data
        X.col(j) = X.col(j).unaryExpr(std_op).eval();
    }

    return X;
}

template <typename Derived>
void
normalize_columns(Eigen::MatrixBase<Derived> &_mat)
{
    using Index = typename Derived::Index;
    using Scalar = typename Derived::Scalar;

    Derived &mat = _mat.derived();
    const Scalar eps = 1e-8;

    for (Index c = 0; c < mat.cols(); ++c) {
        const Scalar denom = std::max(mat.col(c).norm(), eps);
        mat.col(c) /= denom;
    }
}

template <typename Derived>
void
normalize_columns(Eigen::SparseMatrixBase<Derived> &_mat)
{
    using Index = typename Derived::Index;
    using Scalar = typename Derived::Scalar;

    Derived &mat = _mat.derived();
    const Scalar eps = 1e-8;

    std::vector<Scalar> col_norm(mat.cols());
    std::fill(col_norm.begin(), col_norm.end(), 0.0);

    for (Index k = 0; k < mat.outerSize(); ++k) {
        for (typename Derived::InnerIterator it(mat, k); it; ++it) {
            const Scalar x = it.value();
            col_norm[it.col()] += x * x;
        }
    }

    for (Index k = 0; k < mat.outerSize(); ++k) {
        for (typename Derived::InnerIterator it(mat, k); it; ++it) {
            const Scalar x = it.value();
            const Scalar denom = std::sqrt(col_norm[it.col()]);
            it.valueRef() = x / std::max(denom, eps);
        }
    }
}

////////////////////////////////////////////////////////////////
template <typename Derived>
void
setConstant(Eigen::SparseMatrixBase<Derived> &mat,
            const typename Derived::Scalar val)
{
    using Scalar = typename Derived::Scalar;
    auto fill_const = [val](const Scalar &x) { return val; };
    Derived &Mat = mat.derived();
    Mat = Mat.unaryExpr(fill_const);
}

template <typename Derived>
void
setConstant(Eigen::MatrixBase<Derived> &mat, const typename Derived::Scalar val)
{
    Derived &Mat = mat.derived();
    Mat.setConstant(val);
}

template <typename T>
struct running_stat_t {
    using Scalar = typename T::Scalar;
    using Index = typename T::Index;

    explicit running_stat_t(const Index _d1, const Index _d2)
        : d1{ _d1 }
        , d2{ _d2 }
    {
        Cum.resize(d1, d2);
        SqCum.resize(d1, d2);
        Mean.resize(d1, d2);
        Var.resize(d1, d2);
        reset();
    }

    void reset()
    {
        setConstant(SqCum, 0.0);
        setConstant(Cum, 0.0);
        setConstant(Mean, 0.0);
        setConstant(Var, 0.0);
        n = 0.0;
    }

    template <typename Derived>
    void operator()(const Eigen::MatrixBase<Derived> &X)
    {
        Cum += X;
        SqCum += X.cwiseProduct(X);
        n += 1.0;
    }

    template <typename Derived>
    void operator()(const Eigen::SparseMatrixBase<Derived> &X)
    {
        Cum += X;
        SqCum += X.cwiseProduct(X);
        n += 1.0;
    }

    const T &mean()
    {
        if (n > 0) {
            Mean = Cum / n;
        }
        return Mean;
    }

    const T &var()
    {
        if (n > 1.) {
            Mean = Cum / n;
            Var = SqCum / (n - 1.) - Mean.cwiseProduct(Mean) * n / (n - 1.);
            Var = Var.unaryExpr(clamp_zero_op);
        } else {
            Var.setZero();
        }
        return Var;
    }

    const Index d1;
    const Index d2;

    struct clamp_zero_op_t {
        const Scalar operator()(const Scalar &x) const
        {
            return x < .0 ? 0. : x;
        }
    } clamp_zero_op;

    T Cum;
    T SqCum;
    T Mean;
    T Var;
    Scalar n;
};

template <typename T>
struct is_obs_op {
    using Scalar = typename T::Scalar;
    const Scalar operator()(const Scalar &x) const
    {
        return std::isfinite(x) ? one_val : zero_val;
    }
    static constexpr Scalar one_val = 1.0;
    static constexpr Scalar zero_val = 0.0;
};

template <typename T>
struct add_pseudo_op {
    using Scalar = typename T::Scalar;
    explicit add_pseudo_op(const Scalar pseudo_val)
        : val(pseudo_val)
    {
    }
    const Scalar operator()(const Scalar &x) const { return x + val; }
    const Scalar val;
};

template <typename T1, typename T2, typename Ret>
void
XY_nobs(const Eigen::MatrixBase<T1> &X,
        const Eigen::MatrixBase<T2> &Y,
        Eigen::MatrixBase<Ret> &ret,
        const typename Ret::Scalar pseudo = 1.0)
{
    is_obs_op<T1> op1;
    is_obs_op<T2> op2;
    ret.derived() = (X.unaryExpr(op1) * Y.unaryExpr(op2)).array() + pseudo;
}

template <typename T1, typename T2, typename Ret>
void
XY_nobs(const Eigen::MatrixBase<T1> &X,
        const Eigen::MatrixBase<T2> &Y,
        Eigen::SparseMatrixBase<Ret> &ret,
        const typename Ret::Scalar pseudo = 1.0)
{
    is_obs_op<T1> op1;
    is_obs_op<T2> op2;
    add_pseudo_op<Ret> op_add(pseudo);

    times_set(X.unaryExpr(op1), Y.unaryExpr(op2), ret);
    ret.derived() = ret.unaryExpr(op_add);
}

template <typename T1, typename T2, typename T3, typename Ret>
void
XYZ_nobs(const Eigen::MatrixBase<T1> &X,
         const Eigen::MatrixBase<T2> &Y,
         const Eigen::MatrixBase<T3> &Z,
         Eigen::MatrixBase<Ret> &ret,
         const typename Ret::Scalar pseudo = 1.0)
{
    is_obs_op<T1> op1;
    is_obs_op<T2> op2;
    is_obs_op<T3> op3;

    ret.derived() =
        (X.unaryExpr(op1) * Y.unaryExpr(op2) * Z.unaryExpr(op3)).array() +
        pseudo;
}

template <typename T1, typename T2, typename T3, typename Ret>
void
XYZ_nobs(const Eigen::MatrixBase<T1> &X,
         const Eigen::MatrixBase<T2> &Y,
         const Eigen::MatrixBase<T3> &Z,
         Eigen::SparseMatrixBase<Ret> &ret,
         const typename Ret::Scalar pseudo = 1.0)
{
    is_obs_op<T1> op1;
    is_obs_op<T2> op2;
    is_obs_op<T3> op3;
    add_pseudo_op<Ret> op_add(pseudo);

    auto YZ = (Y.unaryExpr(op2) * Z.unaryExpr(op3)).eval();
    times_set(X.unaryExpr(op1), YZ, ret);
    ret.derived() = ret.unaryExpr(op_add);
}

template <typename T1, typename T2, typename T3, typename Ret>
void
XYZ_nobs(const Eigen::MatrixBase<T1> &X,
         const Eigen::MatrixBase<T2> &Y,
         const Eigen::SparseMatrixBase<T3> &Z,
         Eigen::MatrixBase<Ret> &ret,
         const typename Ret::Scalar pseudo = 1.0)
{
    is_obs_op<T1> op1;
    is_obs_op<T2> op2;
    is_obs_op<T3> op3;
    add_pseudo_op<Ret> op_add(pseudo);

    auto YZ = (Y.unaryExpr(op2) * Z.unaryExpr(op3)).eval();
    times_set(X.unaryExpr(op1), YZ, ret);
    ret.derived() = ret.unaryExpr(op_add);
}

template <typename T1, typename T2, typename T3, typename Ret>
void
XYZ_nobs(const Eigen::MatrixBase<T1> &X,
         const Eigen::MatrixBase<T2> &Y,
         const Eigen::SparseMatrixBase<T3> &Z,
         Eigen::SparseMatrixBase<Ret> &ret,
         const typename Ret::Scalar pseudo = 1.0)
{
    is_obs_op<T1> op1;
    is_obs_op<T2> op2;
    is_obs_op<T3> op3;
    add_pseudo_op<Ret> op_add(pseudo);

    auto YZ = (Y.unaryExpr(op2) * Z.unaryExpr(op3)).eval();
    times_set(X.unaryExpr(op1), YZ, ret);
    ret.derived() = ret.unaryExpr(op_add);
}

template <typename T1, typename T2, typename T3, typename Ret>
void
XYZ_nobs(const Eigen::SparseMatrixBase<T1> &X,
         const Eigen::SparseMatrixBase<T2> &Y,
         const Eigen::MatrixBase<T3> &Z,
         Eigen::SparseMatrixBase<Ret> &ret,
         const typename Ret::Scalar pseudo = 1.0)
{
    is_obs_op<T1> op1;
    is_obs_op<T2> op2;
    is_obs_op<T3> op3;
    add_pseudo_op<Ret> op_add(pseudo);

    auto YZ = (Y.unaryExpr(op2) * Z.unaryExpr(op3)).eval();
    times_set(X.unaryExpr(op1), YZ, ret);
    ret.derived() = ret.unaryExpr(op_add);
}

template <typename T1, typename T2, typename T3, typename Ret>
void
XYZ_nobs(const Eigen::MatrixBase<T1> &X,
         const Eigen::SparseMatrixBase<T2> &Y,
         const Eigen::MatrixBase<T3> &Z,
         Eigen::SparseMatrixBase<Ret> &ret,
         const typename Ret::Scalar pseudo = 1.0)
{
    is_obs_op<T1> op1;
    is_obs_op<T2> op2;
    is_obs_op<T3> op3;
    add_pseudo_op<Ret> op_add(pseudo);

    auto YZ = (Y.unaryExpr(op2) * Z.unaryExpr(op3)).eval();
    times_set(X.unaryExpr(op1), YZ, ret);
    ret.derived() = ret.unaryExpr(op_add);
}

////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename Ret>
void
XtY_nobs(const Eigen::MatrixBase<T1> &X,
         const Eigen::MatrixBase<T2> &Y,
         Eigen::MatrixBase<Ret> &ret,
         const typename Ret::Scalar pseudo = 1.0)
{
    XY_nobs(X.transpose(), Y, ret, pseudo);
}

template <typename T1, typename T2, typename Ret>
void
XtY_nobs(const Eigen::MatrixBase<T1> &X,
         const Eigen::MatrixBase<T2> &Y,
         Eigen::SparseMatrixBase<Ret> &ret,
         const typename Ret::Scalar pseudo = 1.0)
{
    XY_nobs(X.transpose(), Y, ret, pseudo);
}

#endif
