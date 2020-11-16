#ifndef NET_UTIL_HH_
#define NET_UTIL_HH_

#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <vector>

#include "utils/util.hh"

struct network_component_t {
    using Scalar = float;
    using sp_mat_t = Eigen::SparseMatrix<Scalar>;
    using Index = sp_mat_t::Index;

    std::vector<std::string> index2vertex;
    sp_mat_t A;                                 // vertex x vertex
    sp_mat_t Mleft;                             // left vertex x edge
    sp_mat_t Mright;                            // right vertex x edge
    std::vector<std::pair<Index, Index>> Edges; // edges (i,j)
    std::vector<Index> colors;                  // edge colors
};

std::vector<std::shared_ptr<network_component_t>>
read_network_data(const std::string data_file,
                  const std::string color_file,
                  const bool,
                  const double);

template <typename Derived, typename Pair>
int construct_edge_incidence(const Eigen::SparseMatrixBase<Derived> &A,
                             Eigen::SparseMatrixBase<Derived> &Mleft,
                             Eigen::SparseMatrixBase<Derived> &Mright,
                             std::vector<Pair> &edges);

template <typename Data, typename Str2Int, typename Derived>
void read_sparse_pairs(const Data &data,
                       const Str2Int &vertex2index,
                       Eigen::SparseMatrixBase<Derived> &Amat);

template <typename Data, typename Str2Int, typename Graph>
void build_boost_graph(const Data &data, const Str2Int &vertex2index, Graph &G);

template <typename Graph, typename Scalar>
void prune_uninformative_edges(const Graph &gIn, Graph &gOut, const Scalar);

template <typename Data, typename Str2Int, typename Int2Str>
void build_vertex2index(const Data &data,
                        Str2Int &vertex2index,
                        Int2Str &index2vertex);

template <typename Derived, typename OtherDerived, typename Pair>
int construct_incidence_matrices(const Eigen::SparseMatrixBase<Derived> &A,
                                 Eigen::SparseMatrixBase<OtherDerived> &mleft,
                                 Eigen::SparseMatrixBase<OtherDerived> &mright,
                                 std::vector<Pair> &edges);

#include "utils/net_util_impl.hh"

#endif
