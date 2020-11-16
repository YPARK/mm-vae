#include <tuple>
#include <type_traits>
#include <utility>

#ifndef YPP_TUPLE_UTIL_HH_
#define YPP_TUPLE_UTIL_HH_

//////////////////////////////////////////////////////////////////
// apply function, one by one each, to each element of tuple    //
// e.g.,						        //
// func_apply(						        //
//   [](auto &&x) {					        //
//     std::cout << x.name << std::endl;		        //
//   },							        //
// std::make_tuple(eta_mean1, eta_mean2, eta_mean3, eta_var1)); //
//////////////////////////////////////////////////////////////////

template <typename Func, typename... Ts>
void func_apply(Func &&func, std::tuple<Ts...> &&tup);

/////////////////////////////////////////////////////////////////
// create an arbitrary size tuples with lambda		       //
// e.g.,						       //
//   create_tuple<10>([](std::size_t j) { return obj.at(j); }) //
/////////////////////////////////////////////////////////////////

template <std::size_t N, typename Func>
auto create_tuple(Func func);

/////////////////////
// implementations //
/////////////////////

template <typename Func, std::size_t... Is>
auto
create_tuple_impl(Func func, std::index_sequence<Is...>)
{
    return std::make_tuple(func(Is)...);
}

template <std::size_t N, typename Func>
auto
create_tuple(Func func)
{
    return create_tuple_impl(func, std::make_index_sequence<N>{});
}

////////////////////////////////////////////////////////////////
// apply function, one by one each, to each element of tuple
// 1. recurse
template <typename Func, typename Tuple, unsigned N>
struct func_apply_impl_t {
    static void run(Func &&f, Tuple &&tup)
    {
        func_apply_impl_t<Func, Tuple, N - 1>::run(std::forward<Func>(f),
                                                   std::forward<Tuple>(tup));
        std::forward<Func>(f)(std::get<N>(std::forward<Tuple>(tup)));
    }
};

// 2. basecase
template <typename Func, typename Tuple>
struct func_apply_impl_t<Func, Tuple, 0> {
    static void run(Func &&f, Tuple &&tup)
    {
        std::forward<Func>(f)(std::get<0>(std::forward<Tuple>(tup)));
    }
};

template <typename Func, typename... Ts>
void
func_apply(Func &&f, std::tuple<Ts...> &&tup)
{
    using Tuple = std::tuple<Ts...>;
    func_apply_impl_t<Func, Tuple, sizeof...(Ts) - 1>::run(std::forward<Func>(
                                                               f),
                                                           std::forward<Tuple>(
                                                               tup));
}

namespace hash_tuple {

////////////////////
// hash functions //
////////////////////

template <typename T>
struct hash {
    size_t operator()(T const &t) const { return std::hash<T>()(t); }
};

template <typename T>
inline void
hash_combine(std::size_t &seed, const T &val)
{
    std::hash<T> hasher;
    seed ^= hasher(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename S, typename T>
struct hash<std::pair<S, T>> {
    inline std::size_t operator()(const std::pair<S, T> &val) const
    {
        std::size_t seed = 0;
        hash_combine(seed, val.first);
        hash_combine(seed, val.second);
        return seed;
    }
};

template <class... TupleArgs>
struct hash<std::tuple<TupleArgs...>> {
private:
    //  this is a termination condition
    //  N == sizeof...(TupleTypes)
    //
    template <std::size_t Idx, typename... TupleTypes>
    inline typename std::enable_if<Idx == sizeof...(TupleTypes), void>::type
    hash_combine_tup(std::size_t &seed,
                     const std::tuple<TupleTypes...> &tup) const
    {
    }

    //  this is the computation function
    //  continues till condition N < sizeof...(TupleTypes) holds
    //
    template <std::size_t Idx, typename... TupleTypes>
        inline typename std::enable_if <
        Idx<sizeof...(TupleTypes), void>::type
        hash_combine_tup(std::size_t &seed,
                         const std::tuple<TupleTypes...> &tup) const
    {
        hash_combine(seed, std::get<Idx>(tup));

        //  on to next element
        hash_combine_tup<Idx + 1>(seed, tup);
    }

public:
    std::size_t operator()(const std::tuple<TupleArgs...> &tupleValue) const
    {
        std::size_t seed = 0;
        //  begin with the first iteration
        hash_combine_tup<0>(seed, tupleValue);
        return seed;
    }
};

} // hash tuple
#endif
