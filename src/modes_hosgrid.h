#ifndef MODES_HOSGRID_H
#define MODES_HOSGRID_H
#include "AMReX_Gpu.H"
#include "read_modes.h"
#include <fftw3.h>

namespace modes_hosgrid {

enum struct planner_flags { estimate, patient, exhaustive, measure };

void copy_complex(
    const int n0,
    const int n1,
    std::vector<std::complex<double>> complex_vector,
    fftw_complex* ptr);
fftw_complex* allocate_complex(const int n0, const int n1);

void copy_real(
    const int n0, const int n1, std::vector<double> real_vector, double* ptr);
double* allocate_real(const int n0, const int n1);

fftw_plan plan_ifftw(
    const int n0, const int n1, fftw_complex* in, const planner_flags wisdom);

fftw_complex* allocate_plan_copy(
    const int n0,
    const int n1,
    fftw_plan& p,
    std::vector<std::complex<double>> complex_vector);

double* allocate_plan_copy(
    const int n0,
    const int n1,
    std::vector<fftw_plan>& p_vector,
    std::vector<double> real_vector);

fftw_complex* allocate_copy(
    const int n0,
    const int n1,
    std::vector<std::complex<double>> complex_vector);

void plan_ifftw_nwt(
    const int n0,
    const int n1,
    std::vector<fftw_plan>& plan_vector,
    double* in,
    const planner_flags wisdom);

template <typename VT, typename PT>
void populate_hos_eta(
    ReadModes<VT> rm_obj,
    std::vector<fftw_plan> p_vector,
    PT* eta_modes,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_eta);

void populate_hos_eta(
    const int n0,
    const int n1,
    const double dimL,
    std::vector<fftw_plan> p_vector,
    fftw_complex* eta_modes,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_eta);

void populate_hos_ocean_eta_nondim(
    const int n0,
    const int n1,
    fftw_plan p,
    fftw_complex* eta_modes,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_eta);

void populate_hos_eta(
    const int n0,
    const int n1,
    const double dimL,
    std::vector<fftw_plan> p_vector,
    double* eta_modes,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_eta);

void populate_hos_nwt_eta_nondim(
    const int n0,
    const int n1,
    std::vector<fftw_plan> p_vector,
    double* eta_modes,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_eta);

void dimensionalize_eta(
    const double dimL, amrex::Gpu::DeviceVector<amrex::Real>& HOS_eta);

template <typename VT>
void populate_hos_vel(
    ReadModes<VT> rm_obj,
    const double z,
    const double zsl,
    std::vector<VT> mX_vector,
    std::vector<VT> mY_vector,
    std::vector<VT> mZ_vector,
    std::vector<fftw_plan> p_vector,
    fftw_complex* x_modes,
    fftw_complex* y_modes,
    fftw_complex* z_modes,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_u,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_v,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_w,
    int indv_start = 0);

template <typename VT>
void populate_hos_vel(
    ReadModes<VT> rm_obj,
    const double z,
    const double zsl,
    std::vector<VT> mX_vector,
    std::vector<VT> mY_vector,
    std::vector<VT> mZ_vector,
    std::vector<VT> mAdd_vector,
    std::vector<fftw_plan> p_vector,
    double* x_modes,
    double* y_modes,
    double* z_modes,
    double* add_x_modes,
    double* add_y_modes,
    double* add_z_modes,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_u,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_v,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_w,
    int indv_start = 0);

void populate_hos_vel(
    const int n0,
    const int n1,
    const double xlen,
    const double ylen,
    const double depth,
    const double z,
    const double zsl,
    const double dimL,
    const double dimT,
    std::vector<std::complex<double>> mX_vector,
    std::vector<std::complex<double>> mY_vector,
    std::vector<std::complex<double>> mZ_vector,
    std::vector<fftw_plan> p_vector,
    fftw_complex* x_modes,
    fftw_complex* y_modes,
    fftw_complex* z_modes,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_u,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_v,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_w,
    const int indv_start = 0);

void populate_hos_ocean_vel_nondim(
    const int n0,
    const int n1,
    const double nd_xlen,
    const double nd_ylen,
    const double nd_depth,
    const double nd_z,
    std::vector<std::complex<double>> mX_vector,
    std::vector<std::complex<double>> mY_vector,
    std::vector<std::complex<double>> mZ_vector,
    fftw_plan p,
    fftw_complex* x_modes,
    fftw_complex* y_modes,
    fftw_complex* z_modes,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_u,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_v,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_w,
    int indv_start = 0);

void populate_hos_vel(
    const int n0,
    const int n1,
    const int n2,
    const double xlen,
    const double ylen,
    const double z,
    const double zsl,
    const double dimL,
    const double dimT,
    std::vector<double> mX_vector,
    std::vector<double> mY_vector,
    std::vector<double> mZ_vector,
    std::vector<double> mAdd_vector,
    std::vector<fftw_plan> p_vector,
    double* x_modes,
    double* y_modes,
    double* z_modes,
    double* add_x_modes,
    double* add_y_modes,
    double* add_z_modes,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_u,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_v,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_w,
    const int indv_start = 0);

void populate_hos_nwt_vel_nondim(
    const int n0,
    const int n1,
    const double nd_xlen,
    const double nd_ylen,
    const double nd_z,
    std::vector<double> mX_vector,
    std::vector<double> mY_vector,
    std::vector<double> mZ_vector,
    std::vector<fftw_plan> p_vector,
    double* x_modes,
    double* y_modes,
    double* z_modes,
    amrex::Vector<amrex::Real>& nwt_u,
    amrex::Vector<amrex::Real>& nwt_v,
    amrex::Vector<amrex::Real>& nwt_w);

void populate_additional_hos_nwt_vel_nondim(
    const int n0,
    const int n1,
    const int n_add,
    const double nd_xlen,
    const double nd_ylen,
    const double nd_z,
    std::vector<double> add_modes_vec,
    std::vector<fftw_plan> p_vector,
    double* x_modes,
    double* y_modes,
    double* z_modes,
    amrex::Vector<amrex::Real>& nwt_u,
    amrex::Vector<amrex::Real>& nwt_v,
    amrex::Vector<amrex::Real>& nwt_w);

void copy_vel_nondim_to_device(
    amrex::Vector<amrex::Real>& nwt_u,
    amrex::Vector<amrex::Real>& nwt_v,
    amrex::Vector<amrex::Real>& nwt_w,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_u,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_v,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_w,
    const int indv_start);

void dimensionalize_vel(
    const int n0,
    const int n1,
    const double dimL,
    const double dimT,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_u,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_v,
    amrex::Gpu::DeviceVector<amrex::Real>& HOS_w,
    int indv_start = 0);

void do_ifftw(
    const int n0,
    const int n1,
    fftw_plan p,
    fftw_complex* f_in,
    double* sp_out);

void do_ifftw(
    const int n0,
    const int n1,
    const bool cos_y,
    const bool cos_x,
    std::vector<fftw_plan> p_vector,
    double* f_in,
    double* sp_out,
    double* f_work,
    double* sp_work);

void do_ifftw(
    const int n0,
    const bool cos_y,
    std::vector<fftw_plan> p_vector,
    double* f_in,
    double* sp_out,
    double* f_work,
    double* sp_work);

} // namespace modes_hosgrid

#include "modes_hosgrid.tpp"

#endif