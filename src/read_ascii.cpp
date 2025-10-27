#include "read_modes.h"
#include <iterator>

template <>
bool ReadModes<std::complex<double>>::ascii_initialize(bool is_ocean)
{
    std::stringstream fname;
    fname << m_filename;
    std::ifstream is(fname.str());

    bool file_exists = is.good();
    if (!file_exists) {
        return file_exists;
    }

    // Check for is_ocean compatibility

    // Grid2Grid hosOcean.inc, line 270
    double d_n1, d_n2;
    is >> d_n1 >> d_n2 >> dt_out >> T_stop >> xlen >> ylen >> depth >> g >> L >>
        T;

    if (std::abs(T) < 1e-12) {
        std::cout << "Waves2AMR: "
                     "ReadModes<std::complex<double>>::ascii_initialize() \n"
                  << "Time scale read as 0. HOS modes file is not an HOS-Ocean "
                     "output "
                     "and is likely from HOS-NWT instead.\n";
        std::exit(1);
    }

    // xlen, ylen, depth, and g are nondimensionalized
    // L and T are dimensional

    // Convert values
    n1 = (int)d_n1;
    n2 = (int)d_n2;
    f_out = 1.0 / dt_out;

    return file_exists;
}

template <>
bool ReadModes<std::complex<double>>::ascii_read_full(const int itime)
{
    bool eof_not_found = true;
    std::stringstream fname;
    fname << m_filename;
    std::ifstream is(fname.str());
    // Skip previous timesteps
    // -- each entry is complex (2) and 6 vars
    // -- each number takes 18 spaces
    // -- long long is to avoid overflow
    is.ignore(18 * 6 * 2 * (long long)(vec_size * (itime + 1)));
    // Check for eof, exit early if not found
    eof_not_found = eof_not_found && !is.eof();
    if (!eof_not_found) {
        return eof_not_found;
    }
    // Address edge case of itime = -1
    int i1_init = 0;
    if (itime == -1) {
        i1_init = 5;
        is.ignore(18 * 10);
        for (int i1 = 0; i1 < i1_init; ++i1) {
            modeX[i1].real(0.0);
            modeX[i1].imag(0.0);
        }
        // Check for eof, exit early if not found
        eof_not_found = eof_not_found && !is.eof();
        if (!eof_not_found) {
            return eof_not_found;
        }
    }
    // Read modes
    int idx = 0;
    double buf_r, buf_i;
    for (int i2 = 0; i2 < n2; ++i2) {

        for (int i1 = i1_init; i1 < n1o2p1; ++i1) {
            is >> buf_r >> buf_i;
            modeX[idx + i1].real(buf_r);
            modeX[idx + i1].imag(buf_i);
        }
        for (int i1 = 0; i1 < n1o2p1; ++i1) {
            is >> buf_r >> buf_i;
            modeY[idx + i1].real(buf_r);
            modeY[idx + i1].imag(buf_i);
        }
        for (int i1 = 0; i1 < n1o2p1; ++i1) {
            is >> buf_r >> buf_i;
            modeZ[idx + i1].real(buf_r);
            modeZ[idx + i1].imag(buf_i);
        }
        for (int i1 = 0; i1 < n1o2p1; ++i1) {
            is >> buf_r >> buf_i;
            modeT[idx + i1].real(buf_r);
            modeT[idx + i1].imag(buf_i);
        }
        for (int i1 = 0; i1 < n1o2p1; ++i1) {
            is >> buf_r >> buf_i;
            modeFS[idx + i1].real(buf_r);
            modeFS[idx + i1].imag(buf_i);
        }
        for (int i1 = 0; i1 < n1o2p1; ++i1) {
            is >> buf_r >> buf_i;
            modeFST[idx + i1].real(buf_r);
            modeFST[idx + i1].imag(buf_i);
        }
        idx += n1o2p1;
        i1_init = 0;
        // Check for eof, exit early if found
        eof_not_found = eof_not_found && !is.eof();
        if (!eof_not_found) {
            return eof_not_found;
        }
    }
    // Return eof_not_found value, should be true at this point
    return eof_not_found;
}

template <>
bool ReadModes<std::complex<double>>::ascii_read_brief(const int itime)
{
    bool eof_not_found = true;
    std::stringstream fname;
    fname << m_filename;
    std::ifstream is(fname.str());
    // Skip previous timesteps
    // -- each entry is complex (2) and 6 vars
    // -- each number takes 18 spaces
    // -- long long is to avoid overflow
    is.ignore(18 * 6 * 2 * (long long)(vec_size * (itime + 1)));
    // Check for eof, exit early if not found
    eof_not_found = eof_not_found && !is.eof();
    if (!eof_not_found) {
        return eof_not_found;
    }
    // Address edge case of itime = -1
    int i1_init = 0;
    if (itime == -1) {
        i1_init = 5;
        is.ignore(18 * 10);
        for (int i1 = 0; i1 < i1_init; ++i1) {
            modeX[i1].real(0.0);
            modeX[i1].imag(0.0);
        }
        // Check for eof, exit early if not found
        eof_not_found = eof_not_found && !is.eof();
        if (!eof_not_found) {
            return eof_not_found;
        }
    }
    // Read modes
    int idx = 0;
    double buf_r, buf_i;
    for (int i2 = 0; i2 < n2; ++i2) {

        for (int i1 = i1_init; i1 < n1o2p1; ++i1) {
            is >> buf_r >> buf_i;
            modeX[idx + i1].real(buf_r);
            modeX[idx + i1].imag(buf_i);
        }
        for (int i1 = 0; i1 < n1o2p1; ++i1) {
            is >> buf_r >> buf_i;
            modeY[idx + i1].real(buf_r);
            modeY[idx + i1].imag(buf_i);
        }
        for (int i1 = 0; i1 < n1o2p1; ++i1) {
            is >> buf_r >> buf_i;
            modeZ[idx + i1].real(buf_r);
            modeZ[idx + i1].imag(buf_i);
        }
        for (int i1 = 0; i1 < n1o2p1; ++i1) {
            is >> buf_r >> buf_i;
            // Don't need T
        }
        for (int i1 = 0; i1 < n1o2p1; ++i1) {
            is >> buf_r >> buf_i;
            modeFS[idx + i1].real(buf_r);
            modeFS[idx + i1].imag(buf_i);
        }
        for (int i1 = 0; i1 < n1o2p1; ++i1) {
            is >> buf_r >> buf_i;
            // Don't need T
        }
        idx += n1o2p1;
        i1_init = 0;
        // Check for eof, exit early if found
        eof_not_found = eof_not_found && !is.eof();
        if (!eof_not_found) {
            return eof_not_found;
        }
    }
    // Return eof_not_found value, should be true at this point
    return eof_not_found;
}

// Output is a flag where true = successful read, false = eof found
template <>
bool ReadModes<std::complex<double>>::ascii_read(const int itime)
{

    if (modeT.size() == 0) {
        return ascii_read_brief(itime);
    } else {
        return ascii_read_full(itime);
    }
}

// ---------- Above, functions are for HOS-Ocean ----------- //
// ---------- Below, functions are for HOS-NWT   ----------- //

template <>
bool ReadModes<double>::ascii_initialize(bool is_ocean)
{
    std::stringstream fname;
    fname << m_filename;
    std::ifstream is(fname.str());

    bool file_exists = is.good();
    if (!file_exists) {
        return file_exists;
    }

    // Check for is_ocean compatibility

    // Grid2Grid hosNWT.inc, line 297
    double d_n1, d_n2, d_n3, dummy;
    // (yes, these are supposed to be out of order)
    is >> d_n1 >> d_n3 >> d_n2 >> dt_out >> T_stop >> xlen >> ylen >> depth >>
        dummy;

    if (std::abs(dummy) > 1e-12) {
        std::cout
            << "Waves2AMR: ReadModes<double>::ascii_initialize() \n"
            << "Unexpected nonzero parameters found. HOS modes file is not "
               "an HOS-NWT output and is likely from HOS-Ocean instead.\n";
        std::exit(1);
    }

    // xlen, ylen are nondimensionalized
    // depth is not, see lines 264-265
    L = depth;
    T = sqrt(depth / g_const);

    // Set up non-dimensional quantities that will be dimensionalized later
    depth = 1.0;
    g = g_const / L * T * T;

    // Convert values
    n1 = (int)d_n1;
    n2 = (int)d_n2;
    n3 = (int)d_n3;
    f_out = 1.0 / dt_out;

    return file_exists;
}

template <>
bool ReadModes<double>::ascii_read_full(const int itime)
{
    bool eof_not_found = true;
    std::stringstream fname;
    fname << m_filename;
    std::ifstream is(fname.str());
    // For now, assume "extraInterpolationNumber" is 1

    // Skip previous timesteps
    // -- each entry is real and 8 vars
    // -- each number takes 18 spaces
    // -- long long is to avoid overflow
    // Grid2Grid hosNWT.inc, line 425 (recl * nrecl)
    is.ignore(18 * 8 * (long long)(vec_size * (itime + 1)));
    // Check for eof, exit early if not found
    eof_not_found = eof_not_found && !is.eof();
    if (!eof_not_found) {
        return eof_not_found;
    }
    // Address edge case of itime = -1
    int i1_init = 0;
    if (itime == -1) {
        i1_init = 7;
        is.ignore(18 * 8);
        for (int i1 = 0; i1 < i1_init; ++i1) {
            modeX[i1] = 0.0;
        }
        // Check for eof, exit early if not found
        eof_not_found = eof_not_found && !is.eof();
        if (!eof_not_found) {
            return eof_not_found;
        }
    }
    // Read modes
    int idx = 0;
    int idx3 = 0;
    for (int i2 = 0; i2 < n2; ++i2) {

        for (int i1 = i1_init; i1 < n1; ++i1) {
            is >> modeX[idx + i1];
        }
        for (int i1 = 0; i1 < n1; ++i1) {
            is >> modeY[idx + i1];
        }
        for (int i1 = 0; i1 < n1; ++i1) {
            is >> modeZ[idx + i1];
        }
        for (int i1 = 0; i1 < n1; ++i1) {
            is >> modeT[idx + i1];
        }
        for (int i1 = 0; i1 < n1; ++i1) {
            is >> modeFS[idx + i1];
        }
        for (int i1 = 0; i1 < n1; ++i1) {
            is >> modeFST[idx + i1];
        }
        for (int i1 = 0; i1 < n3; ++i1) {
            is >> modeAdd[idx3 + i1];
        }
        for (int i1 = 0; i1 < n3; ++i1) {
            is >> modeAddT[idx3 + i1];
        }
        idx += n1;
        idx3 += n3;
        i1_init = 0;
        // Check for eof, exit early if found
        eof_not_found = eof_not_found && !is.eof();
        if (!eof_not_found) {
            return eof_not_found;
        }
    }
    // Return eof_not_found value, should be true at this point
    return eof_not_found;
}

template <>
bool ReadModes<double>::ascii_read_brief(const int itime)
{
    bool eof_not_found = true;
    std::stringstream fname;
    fname << m_filename;
    std::ifstream is(fname.str());

    // Skip previous timesteps
    // -- each entry is real and 8 vars
    // -- each number takes 18 spaces
    // -- long long is to avoid overflow
    // Grid2Grid hosNWT.inc, line 425 (recl * nrecl)
    is.ignore(18 * 8 * (long long)(vec_size * (itime + 1)));
    // Check for eof, exit early if not found
    eof_not_found = eof_not_found && !is.eof();
    if (!eof_not_found) {
        return eof_not_found;
    }
    // Address edge case of itime = -1
    int i1_init = 0;
    if (itime == -1) {
        i1_init = 7;
        is.ignore(18 * 8);
        for (int i1 = 0; i1 < i1_init; ++i1) {
            modeX[i1] = 0.0;
        }
        // Check for eof, exit early if not found
        eof_not_found = eof_not_found && !is.eof();
        if (!eof_not_found) {
            return eof_not_found;
        }
    }
    // Read modes
    int idx = 0;
    int idx3 = 0;
    double buf_r;
    for (int i2 = 0; i2 < n2; ++i2) {

        for (int i1 = i1_init; i1 < n1; ++i1) {
            is >> modeX[idx + i1];
        }
        for (int i1 = 0; i1 < n1; ++i1) {
            is >> modeY[idx + i1];
        }
        for (int i1 = 0; i1 < n1; ++i1) {
            is >> modeZ[idx + i1];
        }
        for (int i1 = 0; i1 < n1; ++i1) {
            is >> buf_r;
            // Don't need T
        }
        for (int i1 = 0; i1 < n1; ++i1) {
            is >> modeFS[idx + i1];
        }
        for (int i1 = 0; i1 < n1; ++i1) {
            is >> buf_r;
            // Don't need FST
        }
        for (int i1 = 0; i1 < n3; ++i1) {
            is >> modeAdd[idx3 + i1];
        }
        for (int i1 = 0; i1 < n3; ++i1) {
            is >> buf_r;
            // Don't need AddT
        }
        idx += n1;
        idx3 += n3;
        i1_init = 0;
        // Check for eof, exit early if found
        eof_not_found = eof_not_found && !is.eof();
        if (!eof_not_found) {
            return eof_not_found;
        }
    }
    // Return eof_not_found value, should be true at this point
    return eof_not_found;
}

// Output is a flag where true = successful read, false = eof found
template <>
bool ReadModes<double>::ascii_read(const int itime)
{

    if (modeT.size() == 0) {
        return ascii_read_brief(itime);
    } else {
        return ascii_read_full(itime);
    }
}
