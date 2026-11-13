#ifndef READ_MODES_DEFINITIONS_H
#define READ_MODES_DEFINITIONS_H
#include "read_modes.h"

template <typename VT>
ReadModes<VT>::ReadModes(std::string filename, bool is_ocean, bool allmodes)
    : ReadModes()
{
    if (!initialize(filename, is_ocean, allmodes)) {
        std::cout << "ABORT: ReadModes is initializing with a file, but the "
                     "specified file does not exist.\n";
        std::exit(1);
    }
}

template <typename VT>
ReadModes<VT>::ReadModes(
    double dt_out_,
    double T_stop_,
    double xlen_,
    double ylen_,
    double depth_,
    double g_,
    double L_,
    double T_)
    : dt_out(dt_out_)
    , T_stop(T_stop_)
    , xlen(xlen_)
    , ylen(ylen_)
    , depth(depth_)
    , g(g_)
    , L(L_)
    , T(T_)
    , is_init(true)
{
    // ^Manually set metadata for the sake of testing, do other expected steps
    // No treatment of integer dimensions needed at the moment

    // Initialize time index
    itime_now = 0;
    // Initialize output frequency
    f_out = 1.0 / dt_out;

    dimensionalize();
}

// Do-nothing constructor, initializer must be called later
template <typename VT>
ReadModes<VT>::ReadModes() : is_init(false)
{}

template <typename VT>
bool ReadModes<VT>::initialize(
    std::string filename, bool is_ocean, bool allmodes, bool store_modes)
{
    // Check if already initialized
    if (is_init) {
        std::cout << "ABORT: ReadModes has already been initialized, but "
                     "initialize has been called again.\n";
        std::exit(1);
    }
    m_filename = filename;
    is_HOS_Ocean = is_ocean;

    // Set time index value
    itime_now = 0;
    // TODO: Determine filetype

    // Initialize (TODO: according to file type)
    bool file_exists = ascii_initialize(is_HOS_Ocean);
    if (!file_exists) {
        return file_exists;
    }
    set_dimensions(allmodes, store_modes);
    dimensionalize();
    for (const double value : {dt_out, T_stop, xlen, ylen, depth, g, L, T})
        if (!std::isfinite(value))
            throw std::runtime_error("Waves2AMR: nonfinite modes header");
    if (depth <= 0 || xlen <= 0 || ylen < 0 || g <= 0 || T_stop < 0)
        throw std::runtime_error("Waves2AMR: invalid modes header");
    is_init = true;
    return true;
}

template <typename VT>
void ReadModes<VT>::set_dimensions(bool allmodes, bool store_modes)
{
    n1o2p1 = n1 / 2 + 1;
    const auto count =
        static_cast<long long>(n2) * (is_HOS_Ocean ? n1o2p1 : n1);
    const auto additional = is_HOS_Ocean ? 0 : static_cast<long long>(n2) * n3;
    if (n1 <= 0 || n2 <= 0 || (!is_HOS_Ocean && (n3 <= 0 || n3 > n1)) ||
        count > std::numeric_limits<int>::max() ||
        additional > std::numeric_limits<int>::max()) {
        throw std::runtime_error("Waves2AMR: invalid modes dimensions");
    }
    vec_size = static_cast<int>(count);
    vec_add_size = static_cast<int>(additional);
    if (!store_modes) return;
    modeX.resize(vec_size);
    modeY.resize(vec_size);
    modeZ.resize(vec_size);
    modeFS.resize(vec_size);
    if (!is_HOS_Ocean) {
        modeAdd.resize(vec_add_size);
    }
    // These modes are optional
    if (allmodes) {
        modeT.resize(vec_size);
        modeFST.resize(vec_size);
        if (!is_HOS_Ocean) {
            modeAddT.resize(vec_add_size);
        }
    }
}

template <typename VT>
ModesMetadata ReadModes<VT>::metadata() const
{
    if (!is_init) throw std::logic_error("Waves2AMR: reader not initialized");
    return {{n1, n2, n3}, {dt_out, T_stop, xlen, ylen, depth, g, L, T}};
}

template <typename VT>
void ReadModes<VT>::initialize(const ModesMetadata& data, bool is_ocean)
{
    if (is_init)
        throw std::logic_error("Waves2AMR: reader already initialized");
    if (is_ocean != std::is_same_v<VT, std::complex<double>>)
        throw std::invalid_argument(
            "Waves2AMR: mode type and metadata disagree");
    n1 = data.dimensions[0];
    n2 = data.dimensions[1];
    n3 = data.dimensions[2];
    dt_out = data.scalars[0];
    T_stop = data.scalars[1];
    xlen = data.scalars[2];
    ylen = data.scalars[3];
    depth = data.scalars[4];
    g = data.scalars[5];
    L = data.scalars[6];
    T = data.scalars[7];
    for (const auto value : data.scalars)
        if (!std::isfinite(value))
            throw std::invalid_argument("Waves2AMR: nonfinite modes metadata");
    if (dt_out <= 0 || L <= 0 || T <= 0 || depth <= 0 || xlen <= 0 ||
        ylen < 0 || g <= 0 || T_stop < 0)
        throw std::invalid_argument("Waves2AMR: invalid modes metadata");
    f_out = 1. / dt_out;
    is_HOS_Ocean = is_ocean;
    itime_now = 0;
    set_dimensions(false, false);
    is_init = true;
}

template <typename VT>
bool ReadModes<VT>::read_into(
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
    return ascii_read_into(itime, mX, mY, mZ, mFS, mAdd, mT, mFST, mAddT);
}

// Version that uses stored index as guess and increments it
template <typename VT>
int ReadModes<VT>::time2step(const double time)
{
    itime_now = time2step(time, itime_now);
    return itime_now;
}

template <typename VT>
int ReadModes<VT>::time2step(const double time, const int itime_guess)
{
    // Return -1 if time is negative
    if (time < 0.0) return -1;
    // Begin with guess
    int itime = itime_guess;
    // Look for same time or after
    bool done = false;
    while (!done) {
        // Use a tolerance to avoid skipping close matches
        if (itime * dt_out < (time - dt_out * 1e-8)) {
            ++itime;
        } else if ((itime - 1) * dt_out > time) {
            --itime;
        } else {
            done = true;
        }
    }
    return itime;
}

template <typename VT>
void ReadModes<VT>::dimensionalize()
{
    // Dimensionalize read-in nondim quantities
    dt_out *= T;
    f_out /= T;
    T_stop *= T;
    xlen *= L;
    ylen *= L;
    depth *= L;
    g *= L / T / T;
}

template <typename VT>
bool ReadModes<VT>::read_data(double time)
{
    int itime = time2step(time);
    return read_data(itime);
}

template <typename VT>
bool ReadModes<VT>::read_data(int itime)
{
    if (itime == m_cached_step) return true;
    m_cached_step = std::numeric_limits<int>::min();
    const bool available = ascii_read(itime);
    if (available) m_cached_step = itime;
    return available;
}

template <typename VT>
void ReadModes<VT>::output_data(
    std::vector<VT>& v1,
    std::vector<VT>& v2,
    std::vector<VT>& v3,
    std::vector<VT>& v4,
    std::vector<VT>& v5,
    std::vector<VT>& v6)
{
    // Copy class variables to input/output variables
    std::copy(modeX.begin(), modeX.end(), v1.begin());
    std::copy(modeY.begin(), modeY.end(), v2.begin());
    std::copy(modeZ.begin(), modeZ.end(), v3.begin());
    std::copy(modeT.begin(), modeT.end(), v4.begin());
    std::copy(modeFS.begin(), modeFS.end(), v5.begin());
    std::copy(modeFST.begin(), modeFST.end(), v6.begin());
}

template <typename VT>
void ReadModes<VT>::output_data(
    std::vector<VT>& v1,
    std::vector<VT>& v2,
    std::vector<VT>& v3,
    std::vector<VT>& v4)
{
    // Copy class variables to input/output variables
    std::copy(modeX.begin(), modeX.end(), v1.begin());
    std::copy(modeY.begin(), modeY.end(), v2.begin());
    std::copy(modeZ.begin(), modeZ.end(), v3.begin());
    std::copy(modeFS.begin(), modeFS.end(), v4.begin());
}

template <typename VT>
bool ReadModes<VT>::get_data(
    double time,
    std::vector<VT>& mX,
    std::vector<VT>& mY,
    std::vector<VT>& mZ,
    std::vector<VT>& mT,
    std::vector<VT>& mFS,
    std::vector<VT>& mFST)
{
    // Read data
    auto flag = read_data(time);
    // Copy data to output
    if (flag) output_data(mX, mY, mZ, mT, mFS, mFST);
    // Pass read flag (for detecting EOF)
    return flag;
}

template <typename VT>
bool ReadModes<VT>::get_data(
    int itime,
    std::vector<VT>& mX,
    std::vector<VT>& mY,
    std::vector<VT>& mZ,
    std::vector<VT>& mT,
    std::vector<VT>& mFS,
    std::vector<VT>& mFST)
{
    // Read data
    auto flag = read_data(itime);
    // Copy data to output
    if (flag) output_data(mX, mY, mZ, mT, mFS, mFST);
    // Pass read flag (for detecting EOF)
    return flag;
}

template <typename VT>
bool ReadModes<VT>::get_data(
    double time,
    std::vector<VT>& mX,
    std::vector<VT>& mY,
    std::vector<VT>& mZ,
    std::vector<VT>& mFS)
{
    // Read data
    auto flag = read_data(time);
    // Copy data to output
    if (flag) output_data(mX, mY, mZ, mFS);
    // Pass read flag (for detecting EOF)
    return flag;
}

template <typename VT>
bool ReadModes<VT>::get_data(
    int itime,
    std::vector<VT>& mX,
    std::vector<VT>& mY,
    std::vector<VT>& mZ,
    std::vector<VT>& mFS)
{
    // Read data
    auto flag = read_data(itime);
    // Copy data to output
    if (flag) output_data(mX, mY, mZ, mFS);
    // Pass read flag (for detecting EOF)
    return flag;
}

template <typename VT>
void ReadModes<VT>::output_data(
    std::vector<VT>& v1,
    std::vector<VT>& v2,
    std::vector<VT>& v3,
    std::vector<VT>& v4,
    std::vector<VT>& v5,
    std::vector<VT>& v6,
    std::vector<VT>& v7,
    std::vector<VT>& v8)
{
    // Copy class variables to input/output variables
    std::copy(modeX.begin(), modeX.end(), v1.begin());
    std::copy(modeY.begin(), modeY.end(), v2.begin());
    std::copy(modeZ.begin(), modeZ.end(), v3.begin());
    std::copy(modeT.begin(), modeT.end(), v4.begin());
    std::copy(modeFS.begin(), modeFS.end(), v5.begin());
    std::copy(modeFST.begin(), modeFST.end(), v6.begin());
    std::copy(modeAdd.begin(), modeAdd.end(), v7.begin());
    std::copy(modeAddT.begin(), modeAddT.end(), v8.begin());
}

template <typename VT>
void ReadModes<VT>::output_data(
    std::vector<VT>& v1,
    std::vector<VT>& v2,
    std::vector<VT>& v3,
    std::vector<VT>& v4,
    std::vector<VT>& v5)
{
    // Copy class variables to input/output variables
    std::copy(modeX.begin(), modeX.end(), v1.begin());
    std::copy(modeY.begin(), modeY.end(), v2.begin());
    std::copy(modeZ.begin(), modeZ.end(), v3.begin());
    std::copy(modeFS.begin(), modeFS.end(), v4.begin());
    std::copy(modeAdd.begin(), modeAdd.end(), v5.begin());
}

template <typename VT>
bool ReadModes<VT>::get_data(
    double time,
    std::vector<VT>& mX,
    std::vector<VT>& mY,
    std::vector<VT>& mZ,
    std::vector<VT>& mT,
    std::vector<VT>& mFS,
    std::vector<VT>& mFST,
    std::vector<VT>& mAdd,
    std::vector<VT>& mAddT)
{
    // Read data
    auto flag = read_data(time);
    // Copy data to output
    if (flag) output_data(mX, mY, mZ, mT, mFS, mFST, mAdd, mAddT);
    // Pass read flag (for detecting EOF)
    return flag;
}

template <typename VT>
bool ReadModes<VT>::get_data(
    int itime,
    std::vector<VT>& mX,
    std::vector<VT>& mY,
    std::vector<VT>& mZ,
    std::vector<VT>& mT,
    std::vector<VT>& mFS,
    std::vector<VT>& mFST,
    std::vector<VT>& mAdd,
    std::vector<VT>& mAddT)
{
    // Read data
    auto flag = read_data(itime);
    // Copy data to output
    if (flag) output_data(mX, mY, mZ, mT, mFS, mFST, mAdd, mAddT);
    // Pass read flag (for detecting EOF)
    return flag;
}

template <typename VT>
bool ReadModes<VT>::get_data(
    double time,
    std::vector<VT>& mX,
    std::vector<VT>& mY,
    std::vector<VT>& mZ,
    std::vector<VT>& mFS,
    std::vector<VT>& mAdd)
{
    // Read data
    auto flag = read_data(time);
    // Copy data to output
    if (flag) output_data(mX, mY, mZ, mFS, mAdd);
    // Pass read flag (for detecting EOF)
    return flag;
}

template <typename VT>
bool ReadModes<VT>::get_data(
    int itime,
    std::vector<VT>& mX,
    std::vector<VT>& mY,
    std::vector<VT>& mZ,
    std::vector<VT>& mFS,
    std::vector<VT>& mAdd)
{
    // Read data
    auto flag = read_data(itime);
    // Copy data to output
    if (flag) output_data(mX, mY, mZ, mFS, mAdd);
    // Pass read flag (for detecting EOF)
    return flag;
}

template <typename VT>
void ReadModes<VT>::print_file_constants()
{
    std::cout << "f_out " << f_out << " T " << T << " T_stop " << T_stop
              << std::endl;
    std::cout << "n1 " << n1 << " n2 " << n2;
    if (n3 > 0) {
        std::cout << " n3 " << n3;
    }
    std::cout << std::endl;
    std::cout << "xlen " << xlen << " ylen " << ylen << std::endl;
    std::cout << "depth " << depth << " g " << g << " L " << L << std::endl;
}

#endif
