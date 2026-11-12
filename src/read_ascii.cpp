#include "read_modes.h"
#include <algorithm>
#include <cmath>

namespace {
std::shared_ptr<std::ifstream>
open_modes(const std::string& filename, std::streamoff& bytes)
{
    auto stream = std::make_shared<std::ifstream>(filename, std::ios::binary);
    if (!*stream) return nullptr;
    stream->seekg(0, std::ios::end);
    bytes = stream->tellg();
    stream->seekg(0);
    if (!*stream)
        throw std::runtime_error("Waves2AMR: cannot seek " + filename);
    return stream;
}

void read_value(std::istream& stream, double& value)
{
    stream >> value;
    if (!stream || !std::isfinite(value))
        throw std::runtime_error("Waves2AMR: malformed modes value");
}

void read_value(std::istream& stream, std::complex<double>& value)
{
    double real, imag;
    read_value(stream, real);
    read_value(stream, imag);
    value = {real, imag};
}

int dimension(double value)
{
    if (!std::isfinite(value) || value <= 0 ||
        value > std::numeric_limits<int>::max() || std::floor(value) != value)
        throw std::runtime_error("Waves2AMR: invalid modes dimension");
    return static_cast<int>(value);
}
} // namespace

template <>
bool ReadModes<std::complex<double>>::ascii_initialize(bool is_ocean)
{
    if (!is_ocean)
        throw std::invalid_argument(
            "Waves2AMR: complex reader requires HOS-Ocean");
    m_stream = open_modes(m_filename, m_file_bytes);
    if (!m_stream) return false;
    double d_n1, d_n2;
    auto& stream = *m_stream;
    stream >> d_n1 >> d_n2 >> dt_out >> T_stop >> xlen >> ylen >> depth >> g >>
        L >> T;
    if (!stream || !std::isfinite(T) || T <= 0 || !std::isfinite(dt_out) ||
        dt_out <= 0 || !std::isfinite(L) || L <= 0)
        throw std::runtime_error("Waves2AMR: invalid HOS-Ocean header");
    n1 = dimension(d_n1);
    n2 = dimension(d_n2);
    f_out = 1. / dt_out;
    return true;
}

template <>
bool ReadModes<double>::ascii_initialize(bool is_ocean)
{
    if (is_ocean)
        throw std::invalid_argument("Waves2AMR: real reader requires HOS-NWT");
    m_stream = open_modes(m_filename, m_file_bytes);
    if (!m_stream) return false;
    double d_n1, d_n2, d_n3, dummy;
    auto& stream = *m_stream;
    stream >> d_n1 >> d_n3 >> d_n2 >> dt_out >> T_stop >> xlen >> ylen >>
        depth >> dummy;
    if (!stream || !std::isfinite(dummy) || std::abs(dummy) > 1e-12 ||
        !std::isfinite(depth) || depth <= 0 || !std::isfinite(dt_out) ||
        dt_out <= 0)
        throw std::runtime_error("Waves2AMR: invalid HOS-NWT header");
    n1 = dimension(d_n1);
    n2 = dimension(d_n2);
    n3 = dimension(d_n3);
    L = depth;
    T = std::sqrt(depth / g_const);
    depth = 1.;
    g = g_const / L * T * T;
    f_out = 1. / dt_out;
    return true;
}

template <typename VT>
bool ReadModes<VT>::ascii_read_into(
    int itime,
    std::vector<VT>& mX,
    std::vector<VT>& mY,
    std::vector<VT>& mZ,
    std::vector<VT>& mFS,
    std::vector<VT>* mAdd,
    std::vector<VT>* mT,
    std::vector<VT>* mFST,
    std::vector<VT>* mAddT)
{
    if (!is_init || !m_stream)
        throw std::logic_error("Waves2AMR: modes reader has no open file");
    if (itime < -1)
        throw std::invalid_argument("Waves2AMR: modes timestep must be >= -1");
    const std::streamoff stride = 18LL * (is_HOS_Ocean ? 12 : 8) * vec_size;
    const auto record = static_cast<std::streamoff>(itime) + 1;
    if (record > std::numeric_limits<std::streamoff>::max() / stride)
        throw std::overflow_error("Waves2AMR: modes record offset overflow");
    auto position = record * stride;
    if (position >= m_file_bytes) return false;
    if (m_file_bytes - position < stride)
        throw std::runtime_error(
            "Waves2AMR: truncated modes record in " + m_filename);
    const int width = is_HOS_Ocean ? n1o2p1 : n1;
    const int components = is_HOS_Ocean ? 2 : 1;
    const int initial = itime == -1 ? (is_HOS_Ocean ? 5 : 7) : 0;
    if (initial > width || (!is_HOS_Ocean && !mAdd))
        throw std::invalid_argument(
            "Waves2AMR: incompatible mode destinations");
    for (auto* modes : {&mX, &mY, &mZ, &mFS, mT, mFST})
        if (modes) modes->resize(vec_size);
    for (auto* modes : {mAdd, mAddT})
        if (modes) modes->resize(vec_add_size);

    auto& stream = *m_stream;
    stream.clear();
    if (initial) {
        position += 18 * (is_HOS_Ocean ? 10 : 8);
        std::fill_n(mX.begin(), initial, VT{});
    }
    if (initial && !is_HOS_Ocean) {
        // The legacy NWT header overlaps X with a one-slot displacement.
        // Preserve its sequential whitespace handling across padded blocks.
        stream.seekg(position);
        auto values = [&](std::vector<VT>* output, int begin, int count) {
            for (int i = 0; i < count; ++i) {
                VT value;
                read_value(stream, value);
                if (output) (*output)[begin + i] = value;
            }
        };
        for (int row = 0; row < n2; ++row) {
            const int begin = row * width;
            const int skip = row == 0 ? initial : 0;
            values(&mX, begin + skip, width - skip);
            values(&mY, begin, width);
            values(&mZ, begin, width);
            values(mT, begin, width);
            values(&mFS, begin, width);
            values(mFST, begin, width);
            values(mAdd, row * n3, n3);
            values(mAddT, row * n3, n3);
        }
        ++m_records_read;
        return true;
    }
    // HOS direct-access records use 18-byte scalar slots, including trailing
    // whitespace. Read retained blocks only; skip derivatives by byte offset.
    auto block = [&](std::vector<VT>* output, int begin, int count) {
        const auto bytes = 18LL * components * count;
        if (output && count) {
            stream.seekg(position);
            std::string text(static_cast<std::size_t>(bytes), ' ');
            stream.read(&text[0], bytes);
            if (!stream)
                throw std::runtime_error(
                    "Waves2AMR: modes I/O error in " + m_filename);
            std::istringstream values(text);
            for (int i = 0; i < count; ++i)
                read_value(values, (*output)[begin + i]);
            values >> std::ws;
            if (!values.eof())
                throw std::runtime_error(
                    "Waves2AMR: invalid fixed-width modes block");
        }
        position += bytes;
    };
    for (int row = 0; row < n2; ++row) {
        const int begin = row * width;
        const int skip = row == 0 ? initial : 0;
        block(&mX, begin + skip, width - skip);
        block(&mY, begin, width);
        block(&mZ, begin, width);
        block(mT, begin, width);
        block(&mFS, begin, width);
        block(mFST, begin, width);
        if (!is_HOS_Ocean) {
            block(mAdd, row * n3, n3);
            position += 18LL * (n1 - n3);
            block(mAddT, row * n3, n3);
            position += 18LL * (n1 - n3);
        }
    }
    ++m_records_read;
    return true;
}

template <typename VT>
bool ReadModes<VT>::ascii_read_full(int itime)
{
    return ascii_read_into(
        itime, modeX, modeY, modeZ, modeFS, is_HOS_Ocean ? nullptr : &modeAdd,
        &modeT, &modeFST, is_HOS_Ocean ? nullptr : &modeAddT);
}

template <typename VT>
bool ReadModes<VT>::ascii_read_brief(int itime)
{
    return read_into(
        itime, modeX, modeY, modeZ, modeFS, is_HOS_Ocean ? nullptr : &modeAdd);
}

template <typename VT>
bool ReadModes<VT>::ascii_read(int itime)
{
    return modeT.empty() ? ascii_read_brief(itime) : ascii_read_full(itime);
}

template bool ReadModes<double>::ascii_read(int);
template bool ReadModes<std::complex<double>>::ascii_read(int);
template bool ReadModes<double>::ascii_read_into(
    int,
    std::vector<double>&,
    std::vector<double>&,
    std::vector<double>&,
    std::vector<double>&,
    std::vector<double>*,
    std::vector<double>*,
    std::vector<double>*,
    std::vector<double>*);
template bool ReadModes<std::complex<double>>::ascii_read_into(
    int,
    std::vector<std::complex<double>>&,
    std::vector<std::complex<double>>&,
    std::vector<std::complex<double>>&,
    std::vector<std::complex<double>>&,
    std::vector<std::complex<double>>*,
    std::vector<std::complex<double>>*,
    std::vector<std::complex<double>>*,
    std::vector<std::complex<double>>*);
