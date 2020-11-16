#include <random>
#include "utils/fastlog.h"
#include "utils/fastexp.h"
#include "utils/fastgamma.h"

#ifndef _UTIL_STAT_HH_
#define _UTIL_STAT_HH_

struct rpois_t {

    rpois_t()
        : unif(0, 1)
    {
    }

    inline float operator()(const float mean)
    {
        if (mean > large_value) {
            return _large(mean);
        } else {
            return _small(mean);
        }
    }

    inline float _small(float mean)
    {
        float L = fasterexp(-mean);
        float p = 1.;
        float ret = 0.;
        do {
            ret++;
            p *= unif(rng);
        } while (p > L);
        ret--;
        return ret;
    }

    inline float _large(float mean)
    {
        float r;
        float x, m;
        float PI = 3.14159265358979;
        float sqrt_mean = std::sqrt(mean);
        float log_mean = fasterlog(mean);
        float g_x;
        float f_m;

        do {
            do {
                x = mean + sqrt_mean * std::tan(PI * (unif(rng) - .5));
            } while (x < 0.);
            g_x = sqrt_mean / (PI * ((x - mean) * (x - mean) + mean));
            m = std::floor(x);
            f_m = fasterexp(m * log_mean - mean - fasterlgamma(m + 1.));
            r = f_m / g_x / 2.4;
        } while (unif(rng) > r);

        return m;
    }

    std::minstd_rand rng{ std::random_device{}() };
    std::uniform_real_distribution<float> unif;
    static constexpr float large_value = 30.;
};

#endif
