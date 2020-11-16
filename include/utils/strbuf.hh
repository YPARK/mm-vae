#include <cctype>
#include <cstdio>
#include <cstdlib>

#include "utils/util.hh"

#ifndef STRBUF_T_HH_
#define STRBUF_T_HH_

////////////////////////////////////////////////////////////////
// C-style string buffer
struct strbuf_t {
    explicit strbuf_t()
    {
        si = -1;
        buf_size = 1024;
        data = new char[buf_size];
        data[0] = '\0';
    }

    ~strbuf_t() { delete[] data; }

    void add(const char c)
    {
        if (!isspace(c)) {
            if ((si + 1) == buf_size)
                double_size();
            data[++si] = c;
            data[si + 1] = '\0';
        }
    }

    size_t size() const { return si + 1; }

    void clear()
    {
        si = -1;
        data[si + 1] = '\0';
    }

    const char *operator()() { return data; }

    template <typename T>
    inline const T get()
    {
        return lexical_cast<T>();
    }

    template <typename T>
    inline const T lexical_cast()
    {
#ifdef DEBUG
        ASSERT(size() > 0, "strbuf.hh: empty strbuf_t");
#endif

        T var = (T)NAN;

        if (!is_data_na()) {
            std::istringstream iss;
            iss.str(data);
            iss >> var;
        }
        return var;
    }

    void take_string(std::string &str)
    {
        str.clear();
        str = data;
    }

    inline const float take_float() const
    {
        // if (is_data_na()) return NAN;
        // return std::atof(data);
        const char *p = data;
        // while (isspace(*p)) ++p;  // ignore white space
        const float TEN = 10.0;
        float r = 0.0;
        bool neg = false;
        if (*p == '-') {
            neg = true;
            ++p;
        }
        // parse >= 1
        while (*p >= '0' && *p <= '9') {
            r = (r * 10.0) + (*p - '0');
            ++p;
        }
        // parse <= 1
        if (*p == '.') {
            float f = 0.0;
            int n = 0;
            ++p;
            while (*p >= '0' && *p <= '9') {
                f = (f * 10.0) + (*p - '0');
                ++p;
                ++n;
            }
            r += f / std::pow(TEN, n);
        }
        // parse the power of 10
        if (*p == 'e' || *p == 'E') {
            ++p;
            float e = 0.0;
            bool neg_exp = false;
            if (*p == '-') {
                neg_exp = true;
                ++p;
            } else if (*p == '+') {
                neg_exp = false;
                ++p;
            }
            while (*p >= '0' && *p <= '9') {
                e = (e * 10.0) + (*p - '0');
                ++p;
            }
            if (neg_exp) {
                e = -e;
            }
            r *= std::pow(TEN, e);
        }
        if (neg) {
            r = -r;
        }
#ifdef DEBUG
        ASSERT(std::abs(std::atof(data) - r) < 1e-4,
               "strbuf.hh: " << data << " vs. " << std::atof(data) << " vs. " << r);
#endif
        return r;
    }

    template <typename T>
    inline const T _take_int() const
    {
        T x = 0;
        bool neg = false;
        const char *p = data;
        // while (isspace(*p)) ++p;  // ignore white space
        if (*p == '-') {
            neg = true;
            ++p;
        }
        while (*p >= '0' && *p <= '9') {
            x = (x * 10) + (*p - '0');
            ++p;
        }
        if (neg) {
            x = -x;
        }

        return x;
    }

    inline const int take_int32() const { return _take_int<int>(); }

    inline const unsigned int take_uint32() const
    {
        return _take_int<unsigned int>();
    }

    inline const std::ptrdiff_t take_uint64() const
    {
        return _take_int<std::ptrdiff_t>();
    }

private:
    bool is_data_na()
    {
        if (size() >= 2) {
            const char c1 = data[0];
            const char c2 = data[1];
            return ((c1 == 'N') & (c2 == 'a')) || ((c1 == 'n') & (c2 == 'a')) ||
                ((c1 == 'N') & (c2 == 'A'));
        }
        return false;
    }

    void double_size()
    {
        char *newdata = new char[2 * buf_size];
        std::copy(data, data + buf_size, newdata);

        buf_size *= 2;
        delete[] data;
        data = newdata;
    }

    size_t si;
    size_t buf_size;
    char *data;
};

#endif
