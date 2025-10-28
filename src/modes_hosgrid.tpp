#include <type_traits>

// Uses ReadModes object directly instead of of separate variables
template <typename VT>
void modes_hosgrid::populate_hos_vel(
    ReadModes<VT> rm_obj,
    const double z,
    const double zsl,
    const std::vector<VT>& mX_vector,
    const std::vector<VT>& mY_vector,
    const std::vector<VT>& mZ_vector,
    const std::vector<fftw_plan>& p_vector,
    fftw_complex* x_modes,
    fftw_complex* y_modes,
    fftw_complex* z_modes,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_u,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_v,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_w,
    int indv_start)
{

    // Get nondimensional velocities
    populate_hos_vel(
        rm_obj.get_first_fft_dimension(), rm_obj.get_second_fft_dimension(),
        rm_obj.get_xlen(), rm_obj.get_ylen(), rm_obj.get_depth(), z, zsl,
        rm_obj.get_L(), rm_obj.get_T(), mX_vector, mY_vector, mZ_vector,
        p_vector, x_modes, y_modes, z_modes, HOS_u, HOS_v, HOS_w, indv_start);
}

template <typename VT>
void modes_hosgrid::populate_hos_vel(
    ReadModes<VT> rm_obj,
    const double z,
    const double zsl,
    const std::vector<VT>& mX_vector,
    const std::vector<VT>& mY_vector,
    const std::vector<VT>& mZ_vector,
    const std::vector<VT>& mAdd_vector,
    const std::vector<fftw_plan>& p_vector,
    double* x_modes,
    double* y_modes,
    double* z_modes,
    double* add_x_modes,
    double* add_y_modes,
    double* add_z_modes,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_u,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_v,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_w,
    int indv_start)
{

    // Get nondimensional velocities
    populate_hos_vel(
        rm_obj.get_first_fft_dimension(), rm_obj.get_second_fft_dimension(),
        rm_obj.get_third_dimension(), rm_obj.get_xlen(), rm_obj.get_ylen(), z,
        zsl, rm_obj.get_L(), rm_obj.get_T(), mX_vector, mY_vector, mZ_vector,
        mAdd_vector, p_vector, x_modes, y_modes, z_modes, add_x_modes,
        add_y_modes, add_z_modes, HOS_u, HOS_v, HOS_w, indv_start);
}

// Uses ReadModes object directly instead of of separate variables
template <typename VT, typename PT>
void modes_hosgrid::populate_hos_eta(
    ReadModes<VT> rm_obj,
    const std::vector<fftw_plan>& p_vector,
    PT* eta_modes,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_eta)
{

    bool types_lengths_match = (std::is_same_v<VT, double>)
                                   ? p_vector.size() > 1
                                   : p_vector.size() == 1;
    if (!types_lengths_match) {
        std::cout << "ABORT: Waves2AMR modes_hosgrid::populate_hos_eta\n"
                  << "       templated type of ReadModes object does not match "
                     "expected length of fftw plan vector.\n"
                  << "       HOS-Ocean and HOS-NWT pathways are likely being "
                     "erroneously mixed.\n";
        std::exit(1);
    }

    // Pass parameters to function via object calls
    populate_hos_eta(
        rm_obj.get_first_fft_dimension(), rm_obj.get_second_fft_dimension(),
        rm_obj.get_L(), p_vector, eta_modes, HOS_eta);
}
