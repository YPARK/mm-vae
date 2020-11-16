#include <iomanip>
#include <iostream>

#ifndef PROGRESS_HH_
#define PROGRESS_HH_

template <typename index_t>
struct progress_bar_t {
    explicit progress_bar_t(const index_t max_iter, const index_t interval)
        : MAX_ITER(max_iter)
        , INTERVAL(interval)
        , MAX_PRINT(max_iter / interval)
    {
        iter = 0;
    }

    void set_zero() { iter = 0; }

    void update() { iter++; }

    template <typename OFS>
    inline void operator()(OFS &ofs)
    {
        if (iter % INTERVAL == 0) {
            ofs << "\r" << std::setw(30) << iter;
            ofs << std::setw(10) << (100 * iter / MAX_ITER) << "%";
            ofs << "\r" << std::flush;
        }
    }

    const index_t MAX_ITER;
    const index_t INTERVAL;
    const index_t MAX_PRINT;

private:
    index_t iter;
};

#endif
