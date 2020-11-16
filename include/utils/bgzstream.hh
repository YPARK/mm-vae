/////////////////////////////////////////////////////////////////
// A streambuffer wrapper for bgizpped file provided by TABIX. //
// This file is just modified from gzstream.hh and gzstream.cc //
// 							       //
// Mon Mar 16 12:53:30 EDT 2020				       //
/////////////////////////////////////////////////////////////////
#include <zlib.h>
#include <cstring>
#include <iostream>
#include "bgzf.h"

#ifndef BGZ_STREAM_HH_
#define BGZ_STREAM_HH_

class bgzf_streambuf : public std::streambuf {

private:
    static const int bufferSize = 47 + 256; // size of data buff
    // totals 512 bytes under g++ for igzstream at the end.

    BGZF *file;              // file handle for compressed file
    char buffer[bufferSize]; // data buffer
    char opened;             // open/close state of stream
    int mode;                // I/O mode

    int flush_buffer();

public:
    bgzf_streambuf()
        : opened(0)
    {
        setp(buffer, buffer + (bufferSize - 1));
        setg(buffer + 4,  // beginning of putback area
             buffer + 4,  // read position
             buffer + 4); // end position
    }
    int is_open() { return opened; }
    bgzf_streambuf *open(const char *name, int open_mode);
    bgzf_streambuf *close();
    ~bgzf_streambuf() { close(); }

    virtual int overflow(int c = EOF);
    virtual int underflow();
    virtual int sync();
};

class bgzf_streambase : virtual public std::ios {
protected:
    bgzf_streambuf buf;

public:
    bgzf_streambase() { init(&buf); }
    bgzf_streambase(const char *name, int open_mode);
    ~bgzf_streambase();
    void open(const char *name, int open_mode);
    void close();
    bgzf_streambuf *rdbuf() { return &buf; }
};

// ----------------------------------------------------------------------------
// User classes. Use ibgzf_stream and obgzf_stream analogously to ifstream and
// ofstream respectively. They read and write files based on the gz*
// function interface of the zlib. Files are compatible with gzip compression.
// ----------------------------------------------------------------------------

class ibgzf_stream : public bgzf_streambase, public std::istream {
public:
    ibgzf_stream()
        : std::istream(&buf)
    {
    }
    ibgzf_stream(const char *name, int open_mode = std::ios::in)
        : bgzf_streambase(name, open_mode)
        , std::istream(&buf)
    {
    }
    bgzf_streambuf *rdbuf() { return bgzf_streambase::rdbuf(); }
    void open(const char *name, int open_mode = std::ios::in)
    {
        bgzf_streambase::open(name, open_mode);
    }
};

class obgzf_stream : public bgzf_streambase, public std::ostream {
public:
    obgzf_stream()
        : std::ostream(&buf)
    {
    }
    obgzf_stream(const char *name, int mode = std::ios::out)
        : bgzf_streambase(name, mode)
        , std::ostream(&buf)
    {
    }
    bgzf_streambuf *rdbuf() { return bgzf_streambase::rdbuf(); }
    void open(const char *name, int open_mode = std::ios::out)
    {
        bgzf_streambase::open(name, open_mode);
    }
};

#endif
