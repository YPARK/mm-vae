#include <algorithm>
#include <functional>
#include <unordered_map>
#include <tuple>
#include <string>
#include <sstream>
#include <vector>

#ifndef STD_UTIL_HH_
#define STD_UTIL_HH_

char *
str2char(const std::string &s)
{
    char *ret = new char[s.size() + 1];
    std::strcpy(ret, s.c_str());
    return ret;
}

std::vector<std::string>
split(const std::string &s, char delim)
{
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> elems;
    while (std::getline(ss, item, delim)) {
        elems.push_back(std::move(item));
    }
    return elems;
}

template <typename Vec>
auto
std_argsort(const Vec &data)
{
    using Index = std::ptrdiff_t;
    std::vector<Index> index(data.size());
    std::iota(std::begin(index), std::end(index), 0);
    std::sort(std::begin(index), std::end(index), [&](Index lhs, Index rhs) {
        return data.at(lhs) > data.at(rhs);
    });
    return index;
}

/**
 * vector -> map: name -> position index
 */
template <typename S, typename I>
std::unordered_map<S, I>
make_position_dict(const std::vector<S> &name_vec)
{

    std::unordered_map<S, I> name_to_id;

    for (I i = 0; i < name_vec.size(); ++i) {
        const S &j = name_vec.at(i);
        name_to_id[j] = i;
    }

    return name_to_id;
}

template <typename S, typename I>
std::tuple<std::vector<I>, std::vector<S>, std::unordered_map<S, I>>
make_indexed_vector(const std::vector<S> &name_vec)
{

    std::unordered_map<S, I> name_to_id;
    std::vector<S> id_to_name;
    std::vector<I> id_vec;
    id_vec.reserve(name_vec.size());

    for (I i = 0; i < name_vec.size(); ++i) {
        const S &ii = name_vec.at(i);
        if (name_to_id.count(ii) == 0) {
            const I j = name_to_id.size();
            name_to_id[ii] = j;
            id_to_name.push_back(ii);
        }
        id_vec.emplace_back(name_to_id.at(ii));
    }

    return std::make_tuple(id_vec, id_to_name, name_to_id);
}

template <typename I>
std::vector<std::vector<I>>
make_index_vec_vec(const std::vector<I> &_id)
{
    using vec_ivec = std::vector<std::vector<I>>;

    const I nn = *std::max_element(_id.begin(), _id.end()) + 1;

    vec_ivec ret(nn, std::vector<I> {});

    for (I i = 0; i < _id.size(); ++i) {
        const I k = _id.at(i);
        ret[k].push_back(i);
    }
    return ret;
}

// template <typename Vec>
// auto
// std_argsort_par(const Vec& data) {
//   using Index = std::ptrdiff_t;
//   std::vector<Index> index(data.size());
//   std::iota(std::begin(index), std::end(index), 0);
//   std::sort(std::execution::par, std::begin(index), std::end(index),
//             [&](Index lhs, Index rhs) { return data.at(lhs) > data.at(rhs);
//             });
//   return index;
// }

#endif
