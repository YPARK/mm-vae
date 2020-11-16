/////////////////////////////////////////////////////////////////
// A streambuffer wrapper for bgizpped file provided by TABIX. //
// The filed is just modified from gzstream.hh and gzstream.cc //
// 							       //
// Mon Mar 16 12:53:30 EDT 2020				       //
/////////////////////////////////////////////////////////////////

#include "bgzstream.hh"

bgzf_streambuf *
bgzf_streambuf::open(const char *name, int open_mode)
{
    if (is_open())
        return (bgzf_streambuf *)0;
    mode = open_mode;
    // no append nor read/write mode
    if ((mode & std::ios::ate) || (mode & std::ios::app) ||
        ((mode & std::ios::in) && (mode & std::ios::out)))
        return (bgzf_streambuf *)0;
    char fmode[10];
    char *fmodeptr = fmode;
    if (mode & std::ios::in)
        *fmodeptr++ = 'r';
    else if (mode & std::ios::out)
        *fmodeptr++ = 'w';
    *fmodeptr++ = 'b';
    *fmodeptr = '\0';
    file = bgzf_open(name, fmode);
    if (file == 0)
        return (bgzf_streambuf *)0;
    opened = 1;
    return this;
}

bgzf_streambuf *
bgzf_streambuf::close()
{
    if (is_open()) {
        sync();
        opened = 0;
        if (bgzf_close(file) == Z_OK)
            return this;
    }
    return (bgzf_streambuf *)0;
}

int
bgzf_streambuf::underflow()
{ // used for input buffer only
    if (gptr() && (gptr() < egptr()))
        return *reinterpret_cast<unsigned char *>(gptr());

    if (!(mode & std::ios::in) || !opened)
        return EOF;
    // Josuttis' implementation of inbuf
    int n_putback = gptr() - eback();
    if (n_putback > 4)
        n_putback = 4;
    memcpy(buffer + (4 - n_putback), gptr() - n_putback, n_putback);

    int num = bgzf_read(file, buffer + 4, bufferSize - 4);
    if (num <= 0) // ERROR or EOF
        return EOF;

    // reset buffer pointers
    setg(buffer + (4 - n_putback), // beginning of putback area
         buffer + 4,               // read position
         buffer + 4 + num);        // end of buffer

    // return next character
    return *reinterpret_cast<unsigned char *>(gptr());
}

int
bgzf_streambuf::flush_buffer()
{
    // Separate the writing of the buffer from overflow() and
    // sync() operation.
    int w = pptr() - pbase();
    if (bgzf_write(file, pbase(), w) != w)
        return EOF;
    pbump(-w);
    return w;
}

int
bgzf_streambuf::overflow(int c)
{ // used for output buffer only
    if (!(mode & std::ios::out) || !opened)
        return EOF;
    if (c != EOF) {
        *pptr() = c;
        pbump(1);
    }
    if (flush_buffer() == EOF)
        return EOF;
    return c;
}

int
bgzf_streambuf::sync()
{
    // Changed to use flush_buffer() instead of overflow( EOF)
    // which caused improper behavior with std::endl and flush(),
    // bug reported by Vincent Ricard.
    if (pptr() && pptr() > pbase()) {
        if (flush_buffer() == EOF)
            return -1;
    }
    return 0;
}

// --------------------------------------
// class bgzf_streambase:
// --------------------------------------

bgzf_streambase::bgzf_streambase(const char *name, int mode)
{
    init(&buf);
    open(name, mode);
}

bgzf_streambase::~bgzf_streambase()
{
    buf.close();
}

void
bgzf_streambase::open(const char *name, int open_mode)
{
    if (!buf.open(name, open_mode))
        clear(rdstate() | std::ios::badbit);
}

void
bgzf_streambase::close()
{
    if (buf.is_open())
        if (!buf.close())
            clear(rdstate() | std::ios::badbit);
}
