#ifndef NET_UTIL_IMPL_HH_
#define NET_UTIL_IMPL_HH_

template <typename Data, typename Str2Int, typename Derived>
void
read_sparse_pairs(const Data &data,
                  const Str2Int &vertex2index,
                  Eigen::SparseMatrixBase<Derived> &Amat)
{
    Derived &out = Amat.derived();

    using Scalar = typename Derived::Scalar;
    using Triplet = Eigen::Triplet<Scalar>;
    using TripletVec = std::vector<Triplet>;
    using Index = typename Eigen::SparseMatrix<Scalar>::Index;

    Index u, v;
    std::string u_str;
    std::string v_str;
    Scalar weight;
    Index umax = 0;
    Index vmax = 0;

    TripletVec tvec;

    for (auto &pp : data) {
        std::tie(u_str, v_str, weight) = pp;
        u = vertex2index.at(u_str);
        v = vertex2index.at(v_str);

        if (u < 0 || v < 0)
            continue;
        if (u > umax)
            umax = u;
        if (v > vmax)
            vmax = v;
        tvec.push_back(Triplet(u, v, weight));
    }

    ASSERT(umax > 0 && vmax > 0, "empty adjacency matrix");

    out.resize(umax + 1, vmax + 1);
    out.setFromTriplets(tvec.begin(), tvec.end());
    out.makeCompressed();
}

template <typename Data, typename Str2Int, typename Int2Str>
void
build_vertex2index(const Data &data,
                   Str2Int &vertex2index,
                   Int2Str &index2vertex)
{
    int pos = 0;

    auto add_vertex = [&pos, &vertex2index, &index2vertex](const auto &v) {
        if (vertex2index.count(v) == 0) {
            vertex2index[v] = pos;
            index2vertex.push_back(v);
            pos++;
        }
    };

    for (auto &pp : data) {
        add_vertex(std::get<0>(pp));
        add_vertex(std::get<1>(pp));
    }
}

template <typename Data, typename Str2Int, typename Graph>
void
build_boost_graph(const Data &data, const Str2Int &vertex2index, Graph &G)
{
    using Vertex = typename boost::graph_traits<Graph>::vertex_descriptor;
    using Edge = typename Graph::edge_descriptor;
    Vertex max_vertex = 0;
    using IndexPair = std::pair<Vertex, Vertex>;
    std::vector<IndexPair> index_pairs;

    auto update_max_vertex =
        [&max_vertex, &vertex2index, &index_pairs](const auto &e) {
            const Vertex u = vertex2index.at(std::get<0>(e));
            const Vertex v = vertex2index.at(std::get<1>(e));
            if (u != v) {
                if (u > max_vertex)
                    max_vertex = u;
                if (v > max_vertex)
                    max_vertex = v;
                index_pairs.push_back(IndexPair(u, v));
            }
        };

    auto add_edge = [&G](const auto &pp) {
        bool has_edge;
        Edge e;
        boost::tie(e, has_edge) = boost::edge(pp.first, pp.second, G);
        if (!has_edge)
            boost::add_edge(pp.first, pp.second, G);
    };

    for_each(data.begin(), data.end(), update_max_vertex);

    // add vertices
    for (auto u = boost::num_vertices(G); u <= max_vertex; ++u)
        boost::add_vertex(G);

    // add edges
    for_each(index_pairs.begin(), index_pairs.end(), add_edge);
}

// remove an edge (u,v) if N(u) & N(v) = empty
// note: we're removing edges so vertices should remain intact
// so, this is different from iterative degree-cutoff
template <typename Graph, typename Scalar>
void
prune_uninformative_edges(const Graph &gIn, Graph &gOut, const Scalar snCutoff)
{
    // add vertices
    for (auto u = boost::num_vertices(gOut); u <= boost::num_vertices(gIn); ++u)
        boost::add_vertex(gOut);

    typename Graph::edge_iterator ei, eEnd;
    for (boost::tie(ei, eEnd) = boost::edges(gIn); ei != eEnd; ++ei) {
        const typename Graph::vertex_descriptor a = boost::source(*ei, gIn);
        const typename Graph::vertex_descriptor b = boost::target(*ei, gIn);

        typename Graph::adjacency_iterator ani, anEnd;
        typename Graph::adjacency_iterator bni, bnEnd;

        // bool has_sn = false;
        Scalar sn = 0.0;
        for (boost::tie(ani, anEnd) = boost::adjacent_vertices(a, gIn);
             ani != anEnd;
             ++ani) {
            const typename Graph::vertex_descriptor an = *ani;
            for (boost::tie(bni, bnEnd) = boost::adjacent_vertices(b, gIn);
                 bni != bnEnd;
                 ++bni) {
                const typename Graph::vertex_descriptor bn = *bni;

                if ((an == bn) && (an != b) && (bn != a)) {
                    if (++sn >= snCutoff)
                        break;
                }
            }
            if (sn >= snCutoff)
                break;
        }

        // if (has_sn) boost::add_edge(a, b, gOut);
        if (sn >= snCutoff)
            boost::add_edge(a, b, gOut);
    }
}

////////////////////////////////////////////////////////////////
std::vector<std::shared_ptr<network_component_t>>
read_network_data(const std::string data_file,
                  const std::string color_file = "",
                  const bool weighted = false,
                  const double snCutoff = 0.0)
{
    using Graph = boost::adjacency_list<boost::vecS,
                                        boost::vecS,
                                        boost::undirectedS,
                                        boost::no_property,
                                        boost::no_property>;

    using Index = network_component_t::Index;
    using Str2Index = boost::unordered_map<std::string, Index>;
    using Index2Str = std::vector<std::string>;

    using WPair = std::tuple<std::string, std::string, double>;
    using WPairVec = std::vector<WPair>;

    ////////////////////////////////////////////////////////////////
    // 0. read full triples
    auto read_triples_stream = [&weighted](auto &ifs) {
        WPairVec ret;
        std::string v1;
        std::string v2;
        std::string ww;
        double weight;
        if (weighted) {
            TLOG("Reading weighted edges");
            while (ifs >> v1 >> v2 >> ww) {
                try {
                    weight = boost::lexical_cast<double>(ww);
                    ret.push_back(WPair(v1, v2, weight));
                } catch (boost::bad_lexical_cast &) {
                    WLOG("Failed to parse weight : " << ww);
                }
            }
        } else {
            weight = 1.0;
            while (ifs >> v1 >> v2) {
                ret.push_back(WPair(v1, v2, weight));
            }
        }
        return ret;
    };

    WPairVec dataTot;

    if (is_gz(data_file)) {
        igzstream ifs(data_file.c_str(), std::ios::in);
        dataTot = read_triples_stream(ifs);
        ifs.close();
    } else {
        std::ifstream ifs(data_file.c_str(), std::ios::in);
        dataTot = read_triples_stream(ifs);
        ifs.close();
    }

    ////////////////////////////////////////////////////////////////
    // 1. read vertex name and construct map: string -> index and index
    // -> string
    Str2Index vertex2indexTot;
    Index2Str index2vertexTot;

    build_vertex2index(dataTot, vertex2indexTot, index2vertexTot);

    // read edge color file if exists
    using SPair = std::pair<std::string, std::string>;
    using Color = boost::unordered_map<SPair, Index>;
    Color color;

    auto read_color = [](auto &ifs) {
        std::string v1;
        std::string v2;
        Index k;
        Color ret;
        while (ifs >> v1 >> v2 >> k) {
            ret[SPair(v1, v2)] = k;
        }
        return ret;
    };

    if (color_file.size() > 0) {
        if (is_gz(color_file)) {
            igzstream ifs(color_file.c_str(), std::ios::in);
            color = read_color(ifs);
            ifs.close();
        } else {
            std::ifstream ifs(color_file.c_str(), std::ios::in);
            color = read_color(ifs);
            ifs.close();
        }
    }

    ////////////////////////////////////////////////////////////////
    // 2. construct (undirected) graph to build vertex name -> component
    // map
    Graph G;
    Graph H;

    build_boost_graph(dataTot, vertex2indexTot, G);

    if (snCutoff > 0.0) {
        Index m = boost::num_edges(G);
        Index m_prev = m + 1; // just bigger
        H = G;
        TLOG("Edge pruning ... number of edges : "
             << m << " shared neighbor >= " << snCutoff);
        while (m_prev > m) {
            m_prev = m;
            Graph H_next;
            prune_uninformative_edges(H, H_next, snCutoff);
            H = H_next;
            m = boost::num_edges(H);
            TLOG("Edge pruning ... number of edges : "
                 << m << " shared neighbor >= " << snCutoff);
        }

    } else {
        H = G;
    }

    ////////////////////////////////////////////////////////////////
    // 3. construct connected component map
    using IndexVec = std::vector<Index>;
    IndexVec membership(boost::num_vertices(H));
    const Index numComp = boost::connected_components(H, &membership[0]);

    TLOG("Found " << numComp << " connected components");

    ////////////////////////////////////////////////////////////////
    // 4. distribute edges by component membership
    using WPairVecVec = std::vector<WPairVec>;

    WPairVecVec dataComp;

    for (auto k = 0; k < numComp; ++k) {
        dataComp.push_back(WPairVec(0));
    }

    using Vertex = typename boost::graph_traits<Graph>::vertex_descriptor;
    using Edge = typename Graph::edge_descriptor;

    std::string v1;
    std::string v2;
    double weight;
    for (const auto &pp : dataTot) {
        std::tie(v1, v2, weight) = pp;
        const Vertex &u1 = vertex2indexTot.at(v1);
        const Vertex &u2 = vertex2indexTot.at(v2);
        bool has_edge;
        Edge e;
        boost::tie(e, has_edge) = boost::edge(u1, u2, H);
        if (has_edge) {
            const auto k = membership.at(u1);
            dataComp[k].push_back(pp);
        }
    }

    auto make_component = [&color](auto &data) {
        std::shared_ptr<network_component_t> ret(new network_component_t);

        using Str2Int = boost::unordered_map<std::string, int>;

        network_component_t &comp = *ret.get();
        Str2Int vertex2index;
        auto &index2vertex = comp.index2vertex;

        // a. vertex name to index map
        build_vertex2index(data, vertex2index, index2vertex);

        // b. add diagonal elements (self loop)
        for (auto vi : vertex2index) {
            data.push_back(WPair(vi.first, vi.first, 1.0));
        }

        // c. build adjacency matrix
        read_sparse_pairs(data, vertex2index, comp.A);
        TLOG("constructed " << comp.A.rows() << " x " << comp.A.cols()
                            << " adjacency matrix");

        // d. build edge incidence matrix
        construct_edge_incidence(comp.A, comp.Mleft, comp.Mright, comp.Edges);
        TLOG("constructed " << comp.Mleft.rows() << " x " << comp.Mleft.cols()
                            << " incidence matrix (" << comp.Mleft.nonZeros()
                            << ")");
        TLOG("constructed " << comp.Mright.rows() << " x " << comp.Mright.cols()
                            << " incidence matrix (" << comp.Mright.nonZeros()
                            << ")");

        // e. assign colors to edges
        const auto numPairs = data.size();
        std::default_random_engine gen;
        std::uniform_int_distribution<Index> unif(0, numPairs - 1);

        comp.colors.clear();
        for (const auto &e : comp.Edges) {
            Index i, j;
            std::tie(i, j) = e;
            auto key = SPair(index2vertex.at(i), index2vertex.at(j));
            Index k = unif(gen);
            if (color.count(key) > 0) {
                k = color.at(key);
            }
            comp.colors.push_back(k);
        }
        return ret;
    };

    std::vector<std::shared_ptr<network_component_t>> ret;

    for (auto &data : dataComp) {
        const auto m = data.size();
        if (m > 0) {
            TLOG("Adding " << m << " edges");
            ret.push_back(make_component(data));
        }
    }

    TLOG("Constructed " << ret.size() << " connected components after pruning");

    return ret;
}

template <typename Derived, typename Pair>
int
construct_edge_incidence(const Eigen::SparseMatrixBase<Derived> &A,
                         Eigen::SparseMatrixBase<Derived> &mleft,
                         Eigen::SparseMatrixBase<Derived> &mright,
                         std::vector<Pair> &edges)
{
    const Derived &Amat = A.derived();
    Derived &Mleft = mleft.derived();
    Derived &Mright = mright.derived();

    using Scalar = typename Derived::Scalar;
    using Index = typename Derived::Index;
    using Triplet = Eigen::Triplet<Scalar>;
    using TripletVec = std::vector<Triplet>;

    TripletVec mvecLeft;
    TripletVec mvecRight;

    edges.clear();
    mvecLeft.reserve(Amat.nonZeros());
    mvecRight.reserve(Amat.nonZeros());
    Index e = 0;
    for (Index j = 0; j < Amat.cols(); ++j) {
        for (typename Derived::InnerIterator it(Amat, j); it; ++it) {
            Index i = it.row();
            Index j = it.col();

            if (i == j)
                continue;

            mvecLeft.push_back(Triplet(i, e, 1.0));
            mvecRight.push_back(Triplet(j, e, 1.0));
            edges.push_back(Pair(i, j));
            e++;
        }
    }

    const Index numEdges = e;
    Mleft.resize(A.rows(), numEdges);
    Mright.resize(A.cols(), numEdges);
    Mleft.setFromTriplets(mvecLeft.begin(), mvecLeft.end());
    Mright.setFromTriplets(mvecRight.begin(), mvecRight.end());
    Mleft.makeCompressed();
    Mright.makeCompressed();

    return EXIT_SUCCESS;
}

#endif
