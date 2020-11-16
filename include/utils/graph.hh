#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <tuple>
#include <unordered_map>

#ifndef UTIL_GRAPH_HH_
#define UTIL_GRAPH_HH_

using UGraph = boost::adjacency_list<boost::listS,       //
                                     boost::vecS,        //
                                     boost::undirectedS, //
                                     boost::no_property, //
                                     boost::no_property>;

template <typename TripleVec, typename T>
void
build_boost_graph(const TripleVec &data, UGraph &G, const T cutoff)
{
    using Vertex = boost::graph_traits<UGraph>::vertex_descriptor;
    using Edge = UGraph::edge_descriptor;

    Vertex max_vertex = 0;
    auto find_max_n = [&max_vertex](const auto &tt) {
        if (std::get<0>(tt) > max_vertex)
            max_vertex = std::get<0>(tt);
        if (std::get<1>(tt) > max_vertex)
            max_vertex = std::get<1>(tt);
    };

    std::for_each(data.begin(), data.end(), find_max_n);
    for (Vertex u = boost::num_vertices(G); u <= max_vertex; ++u)
        boost::add_vertex(G);

    auto add_edge = [&G, &cutoff](const auto &tt) {
        if (std::get<2>(tt) <= cutoff)
            boost::add_edge(std::get<0>(tt), std::get<1>(tt), G);
    };

    std::for_each(data.begin(), data.end(), add_edge);
}

#endif
