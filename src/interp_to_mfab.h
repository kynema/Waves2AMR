#ifndef INTERP_TO_MFAB_H
#define INTERP_TO_MFAB_H
#include "AMReX_MultiFab.H"

namespace interp_to_mfab {

int create_height_vector(
    amrex::Vector<amrex::Real>& hvec,
    const int n,
    const amrex::Real dz0,
    const amrex::Real z_wlev,
    const amrex::Real z_btm,
    int n_above = 1);

int get_local_height_indices(
    amrex::Vector<int>& indvec,
    const amrex::Vector<amrex::Real>& hvec,
    const amrex::Vector<amrex::MultiFab*>& field_fabs,
    const amrex::Vector<amrex::Geometry>& geom);

int get_local_height_indices(
    amrex::Vector<int>& indvec,
    const amrex::Vector<amrex::Real>& hvec,
    const amrex::Vector<amrex::MultiFab*>& field_fabs,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>&
        problo_vec,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& dx_vec);

int get_local_height_indices(
    amrex::Vector<int>& indvec,
    const amrex::Vector<amrex::Real>& hvec,
    const amrex::MultiFab& mfab,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& problo,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx);

// This library assumes height is in z (index of 2)
void get_mfab_mesh_bounds(
    const amrex::MultiFab& mfab,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& problo,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx,
    amrex::Real& mesh_zlo,
    amrex::Real& mesh_zhi,
    int idim = 2);

int local_height_vec_ops(
    amrex::Vector<int>& indvec,
    const amrex::Vector<amrex::Real>& hvec,
    const amrex::Real& mesh_zlo,
    const amrex::Real& mesh_zhi);

int check_lateral_overlap(
    amrex::Real dist,
    int idim,
    const amrex::Vector<amrex::MultiFab*>& field_fabs,
    const amrex::Vector<amrex::Geometry>& geom,
    bool is_hi);

int check_lateral_overlap_lo(
    amrex::Real dist,
    int idim,
    const amrex::Vector<amrex::MultiFab*>& field_fabs,
    const amrex::Vector<amrex::Geometry>& geom);

int check_lateral_overlap_hi(
    amrex::Real dist,
    int idim,
    const amrex::Vector<amrex::MultiFab*>& field_fabs,
    const amrex::Vector<amrex::Geometry>& geom);

int check_lateral_overlap(
    amrex::Real dist,
    int idim,
    const amrex::Vector<amrex::MultiFab*>& field_fabs,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>&
        problo_vec,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>&
        probhi_vec,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& dx_vec,
    bool is_hi);

int check_lateral_overlap_lo(
    amrex::Real dist,
    int idim,
    const amrex::Vector<amrex::MultiFab*>& field_fabs,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>&
        problo_vec,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>&
        probhi_vec,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& dx_vec);

int check_lateral_overlap_hi(
    amrex::Real dist,
    int idim,
    const amrex::Vector<amrex::MultiFab*>& field_fabs,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>&
        problo_vec,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>&
        probhi_vec,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& dx_vec);

void interp_eta_to_levelset_field(
    const int spd_nx,
    const int spd_ny,
    const amrex::Real spd_dx,
    const amrex::Real spd_dy,
    const amrex::Real spd_xlo,
    const amrex::Real spd_ylo,
    const amrex::Real zsl,
    const bool is_periodic,
    const amrex::Gpu::DeviceVector<amrex::Real>& etavec,
    const amrex::Vector<amrex::MultiFab*>& lsfield,
    const amrex::Vector<amrex::Geometry>& geom);

void interp_eta_to_levelset_field(
    const int spd_nx,
    const int spd_ny,
    const amrex::Real spd_dx,
    const amrex::Real spd_dy,
    const amrex::Real spd_xlo,
    const amrex::Real spd_ylo,
    const amrex::Real zsl,
    const bool is_periodic,
    const amrex::Gpu::DeviceVector<amrex::Real>& etavec,
    const amrex::Vector<amrex::MultiFab*>& lsfield,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>&
        problo_vec,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& dx_vec);

void interp_velocity_to_field(
    const int spd_nx,
    const int spd_ny,
    const amrex::Real spd_dx,
    const amrex::Real spd_dy,
    const amrex::Real spd_xlo,
    const amrex::Real spd_ylo,
    const bool is_periodic,
    const amrex::Vector<int>& indvec,
    const amrex::Vector<amrex::Real>& hvec,
    const amrex::Gpu::DeviceVector<amrex::Real>& uvec,
    const amrex::Gpu::DeviceVector<amrex::Real>& vvec,
    const amrex::Gpu::DeviceVector<amrex::Real>& wvec,
    const amrex::Vector<amrex::MultiFab*>& vfield,
    const amrex::Vector<amrex::Geometry>& geom);

void interp_velocity_to_field(
    const int spd_nx,
    const int spd_ny,
    const amrex::Real spd_dx,
    const amrex::Real spd_dy,
    const amrex::Real spd_xlo,
    const amrex::Real spd_ylo,
    const bool is_periodic,
    const amrex::Vector<int>& indvec,
    const amrex::Vector<amrex::Real>& hvec,
    const amrex::Gpu::DeviceVector<amrex::Real>& uvec,
    const amrex::Gpu::DeviceVector<amrex::Real>& vvec,
    const amrex::Gpu::DeviceVector<amrex::Real>& wvec,
    const amrex::Vector<amrex::MultiFab*>& vfield,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>&
        problo_vec,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& dx_vec);

void interp_eta_to_levelset_multifab(
    const int spd_nx,
    const int spd_ny,
    const amrex::Real spd_dx,
    const amrex::Real spd_dy,
    const amrex::Real spd_xlo,
    const amrex::Real spd_ylo,
    const amrex::Real zsl,
    const bool is_periodic,
    const amrex::Gpu::DeviceVector<amrex::Real>& etavec,
    amrex::MultiFab& lsfab,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& problo,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx);

void interp_velocity_to_multifab(
    const int spd_nx,
    const int spd_ny,
    const amrex::Real spd_dx,
    const amrex::Real spd_dy,
    const amrex::Real spd_xlo,
    const amrex::Real spd_ylo,
    const bool is_periodic,
    const amrex::Vector<int>& indvec,
    const amrex::Vector<amrex::Real>& hvec,
    const amrex::Gpu::DeviceVector<amrex::Real>& uvec,
    const amrex::Gpu::DeviceVector<amrex::Real>& vvec,
    const amrex::Gpu::DeviceVector<amrex::Real>& wvec,
    amrex::MultiFab& vfab,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& problo,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx);

void interp_eta_to_levelset_multifab3D(
    const int spd_nx,
    const int spd_ny,
    const amrex::Real spd_dx,
    const amrex::Real spd_dy,
    const amrex::Real spd_xlo,
    const amrex::Real spd_ylo,
    const amrex::Real zsl,
    const bool is_periodic,
    const amrex::Gpu::DeviceVector<amrex::Real>& etavec,
    amrex::MultiFab& lsfab,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& problo,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx);

void interp_velocity_to_multifab3D(
    const int spd_nx,
    const int spd_ny,
    const amrex::Real spd_dx,
    const amrex::Real spd_dy,
    const amrex::Real spd_xlo,
    const amrex::Real spd_ylo,
    const bool is_periodic,
    const amrex::Vector<int>& indvec,
    const amrex::Vector<amrex::Real>& hvec,
    const amrex::Gpu::DeviceVector<amrex::Real>& uvec,
    const amrex::Gpu::DeviceVector<amrex::Real>& vvec,
    const amrex::Gpu::DeviceVector<amrex::Real>& wvec,
    amrex::MultiFab& vfab,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& problo,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx);

void interp_eta_to_levelset_multifab2D(
    const int spd_nx,
    const amrex::Real spd_dx,
    const amrex::Real spd_xlo,
    const amrex::Real zsl,
    const bool is_periodic,
    const amrex::Gpu::DeviceVector<amrex::Real>& etavec,
    amrex::MultiFab& lsfab,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& problo,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx);

void interp_velocity_to_multifab2D(
    const int spd_nx,
    const amrex::Real spd_dx,
    const amrex::Real spd_xlo,
    const bool is_periodic,
    const amrex::Vector<int>& indvec,
    const amrex::Vector<amrex::Real>& hvec,
    const amrex::Gpu::DeviceVector<amrex::Real>& uvec,
    const amrex::Gpu::DeviceVector<amrex::Real>& wvec,
    amrex::MultiFab& vfab,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& problo,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx);

AMREX_GPU_HOST_DEVICE amrex::Real linear_interp(
    const amrex::Real a000,
    const amrex::Real a100,
    const amrex::Real a010,
    const amrex::Real a001,
    const amrex::Real a110,
    const amrex::Real a101,
    const amrex::Real a011,
    const amrex::Real a111,
    const amrex::Real xc,
    const amrex::Real yc,
    const amrex::Real zc,
    const amrex::Real x0,
    const amrex::Real y0,
    const amrex::Real z0,
    const amrex::Real x1,
    const amrex::Real y1,
    const amrex::Real z1);

AMREX_GPU_HOST_DEVICE amrex::Real linear_interp2D(
    const amrex::Real a00,
    const amrex::Real a10,
    const amrex::Real a01,
    const amrex::Real a11,
    const amrex::Real xc,
    const amrex::Real yc,
    const amrex::Real x0,
    const amrex::Real y0,
    const amrex::Real x1,
    const amrex::Real y1);

AMREX_GPU_HOST_DEVICE amrex::Real linear_interp1D(
    const amrex::Real a0,
    const amrex::Real a1,
    const amrex::Real xc,
    const amrex::Real x0,
    const amrex::Real x1);

} // namespace interp_to_mfab

#endif
