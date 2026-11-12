#ifndef READ_MODES_H
#define READ_MODES_H
#include <complex>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <limits>
#include <stdexcept>
#include <cmath>
#include <type_traits>

struct ModesMetadata
{
    std::array<int, 3> dimensions{};
    // Dimensional dt, stop time, x/y lengths, depth, g, length/time scales.
    std::array<double, 8> scalars{};
};

template <typename VT>
class ReadModes
{
public:
    ReadModes(std::string, bool is_ocean = true, bool allmodes = false);

    ReadModes(
        double dt_out_,
        double T_stop_,
        double xlen_,
        double ylen_,
        double depth_,
        double g_,
        double L_,
        double T_);

    ReadModes();

    bool initialize(
        std::string,
        bool is_ocean,
        bool allmodes = false,
        bool store_modes = true);
    void initialize(const ModesMetadata&, bool is_ocean);
    ModesMetadata metadata() const;
    std::size_t records_read() const { return m_records_read; }

    // Direct reads own no coefficient storage or cache; callers may cache their
    // last successful timestep. A failed EOF read leaves destinations
    // untouched.
    bool read_into(
        int itime,
        std::vector<VT>& mX,
        std::vector<VT>& mY,
        std::vector<VT>& mZ,
        std::vector<VT>& mFS,
        std::vector<VT>* mAdd = nullptr);

    void print_file_constants();

    bool read_data(double time);

    bool read_data(int itime);

    void output_data(
        std::vector<VT>& v1,
        std::vector<VT>& v2,
        std::vector<VT>& v3,
        std::vector<VT>& v4,
        std::vector<VT>& v5,
        std::vector<VT>& v6);

    bool get_data(
        double time,
        std::vector<VT>& mX,
        std::vector<VT>& mY,
        std::vector<VT>& mZ,
        std::vector<VT>& mT,
        std::vector<VT>& mFS,
        std::vector<VT>& mFST);

    bool get_data(
        int itime,
        std::vector<VT>& mX,
        std::vector<VT>& mY,
        std::vector<VT>& mZ,
        std::vector<VT>& mT,
        std::vector<VT>& mFS,
        std::vector<VT>& mFST);

    void output_data(
        std::vector<VT>& v1,
        std::vector<VT>& v2,
        std::vector<VT>& v3,
        std::vector<VT>& v4);

    bool get_data(
        double time,
        std::vector<VT>& mX,
        std::vector<VT>& mY,
        std::vector<VT>& mZ,
        std::vector<VT>& mFS);

    bool get_data(
        int itime,
        std::vector<VT>& mX,
        std::vector<VT>& mY,
        std::vector<VT>& mZ,
        std::vector<VT>& mFS);

    void output_data(
        std::vector<VT>& v1,
        std::vector<VT>& v2,
        std::vector<VT>& v3,
        std::vector<VT>& v4,
        std::vector<VT>& v5,
        std::vector<VT>& v6,
        std::vector<VT>& v7,
        std::vector<VT>& v8);

    bool get_data(
        double time,
        std::vector<VT>& mX,
        std::vector<VT>& mY,
        std::vector<VT>& mZ,
        std::vector<VT>& mT,
        std::vector<VT>& mFS,
        std::vector<VT>& mFST,
        std::vector<VT>& mAdd,
        std::vector<VT>& mAddT);

    bool get_data(
        int itime,
        std::vector<VT>& mX,
        std::vector<VT>& mY,
        std::vector<VT>& mZ,
        std::vector<VT>& mT,
        std::vector<VT>& mFS,
        std::vector<VT>& mFST,
        std::vector<VT>& mAdd,
        std::vector<VT>& mAddT);

    void output_data(
        std::vector<VT>& v1,
        std::vector<VT>& v2,
        std::vector<VT>& v3,
        std::vector<VT>& v4,
        std::vector<VT>& v5);

    bool get_data(
        double time,
        std::vector<VT>& mX,
        std::vector<VT>& mY,
        std::vector<VT>& mZ,
        std::vector<VT>& mFS,
        std::vector<VT>& mAdd);

    bool get_data(
        int itime,
        std::vector<VT>& mX,
        std::vector<VT>& mY,
        std::vector<VT>& mZ,
        std::vector<VT>& mFS,
        std::vector<VT>& mAdd);

    // Calculate size of data for each mode variable (# of complex values)
    int get_vector_size() { return vec_size; }
    int get_addl_vector_size() { return vec_add_size; }

    // Convert time to timestep
    int time2step(const double time, const int itime_guess);
    int time2step(const double time);

    // Convert dimensions for fortran reading
    /* bool fortran2cpp() {} */

    // Output functions for use
    int get_first_fft_dimension() { return (from_fortran ? n2 : n1); }
    int get_second_fft_dimension() { return (from_fortran ? n1 : n2); }
    int get_first_spatial_dimension() { return n1; }
    int get_second_spatial_dimension() { return n2; }
    int get_third_dimension() { return n3; }

    // Output functions of parameters
    int get_n1() { return n1; }
    int get_n2() { return n2; }
    int get_n3() { return n3; }
    double get_dtout() { return dt_out; }
    double get_f() { return f_out; }
    double get_Tstop() { return T_stop; }
    double get_xlen() { return xlen; }
    double get_ylen() { return ylen; }
    double get_depth() { return depth; }
    double get_g() { return g; }
    double get_nondim_dtout() { return dt_out / T; }
    double get_nondim_f() { return f_out * T; }
    double get_nondim_Tstop() { return T_stop / T; }
    double get_nondim_xlen() { return xlen / L; }
    double get_nondim_ylen() { return ylen / L; }
    double get_nondim_depth() { return depth / L; }
    double get_nondim_g() { return g / L * T * T; }
    double get_L() { return L; }
    double get_T() { return T; }

private:
    // ASCII functions
    bool ascii_initialize(bool is_ocean);
    bool ascii_read(const int itime);
    bool ascii_read_full(const int itime);
    bool ascii_read_brief(const int itime);
    bool ascii_read_into(
        int itime,
        std::vector<VT>& mX,
        std::vector<VT>& mY,
        std::vector<VT>& mZ,
        std::vector<VT>& mFS,
        std::vector<VT>* mAdd,
        std::vector<VT>* mT,
        std::vector<VT>* mFST,
        std::vector<VT>* mAddT);
    void set_dimensions(bool allmodes, bool store_modes);

    // Dimensionalize read-in quantities
    void dimensionalize();

    // HOS data filename
    std::string m_filename;
    // Copies remain supported by legacy by-value APIs. Every read seeks
    // explicitly, so copied readers can use this handle sequentially.
    std::shared_ptr<std::ifstream> m_stream;
    std::streamoff m_file_bytes{0};
    int m_cached_step{std::numeric_limits<int>::min()};
    std::size_t m_records_read{0};

    // HOS data dimensions
    int n1{0}, n2{0}, n3{-1};
    double dt_out, f_out, T_stop, xlen, ylen, depth, g, L, T;

    // HOS data vectors
    std::vector<VT> modeX, modeY, modeZ, modeT, modeFS, modeFST, modeAdd,
        modeAddT;

    // HOS working dimensions
    int vec_size{0};
    int n1o2p1;          // Ocean only
    int vec_add_size{0}; // NWT only

    // Current time index
    int itime_now;

    // Constants (used in NWT case)
    static constexpr double g_const = 9.81;

    // Relate between format of modes and FFT (only one option now)
    const bool from_fortran = true;

    // Is initialized? for constructor vs initializer functions
    bool is_init{false};

    // Type of mode input and format: HOS-Ocean or HOS-NWT
    bool is_HOS_Ocean{true};
};

#include "read_modes.tpp"

#endif