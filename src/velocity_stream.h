#ifndef VELOCITY_STREAM_H
#define VELOCITY_STREAM_H

#include "AMReX_MultiFab.H"
#include "interpolation_stencil.h"
#include <cstddef>
#include <functional>
#include <vector>

namespace interp_to_mfab {

using VelocityPlaneReader = std::function<void(
    int,
    amrex::Gpu::DeviceVector<amrex::Real>&,
    amrex::Gpu::DeviceVector<amrex::Real>&,
    amrex::Gpu::DeviceVector<amrex::Real>&)>;

using LocalVelocityPlaneReader = std::function<void(
    int,
    const std::vector<int>&,
    const std::vector<int>&,
    amrex::Gpu::DeviceVector<amrex::Real>&,
    amrex::Gpu::DeviceVector<amrex::Real>&,
    amrex::Gpu::DeviceVector<amrex::Real>&)>;

using LocalElevationReader = std::function<void(
    const std::vector<int>&,
    const std::vector<int>&,
    amrex::Gpu::DeviceVector<amrex::Real>&)>;

struct VelocityStreamMemory
{
    std::size_t plane_bytes{0};
    std::size_t stencil_bytes{0};
};

// The reader fills one full FFT plane per component at the requested height
// index. Cubic retains four horizontally interpolated local planes; linear
// retains two sets of raw corners for exact compatibility.
VelocityStreamMemory stream_velocity_to_field(
    int nx,
    int ny,
    amrex::Real sx,
    amrex::Real sy,
    amrex::Real xlo,
    amrex::Real ylo,
    bool periodic,
    const amrex::Vector<int>& indices,
    const amrex::Vector<amrex::Real>& heights,
    const amrex::Vector<amrex::MultiFab*>& fields,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& problo,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& dx,
    const VelocityPlaneReader& read_plane,
    const LocalVelocityPlaneReader& read_local_plane = {},
    Interpolation interpolation = Interpolation::cubic);

void local_eta_to_levelset_field(
    int nx,
    int ny,
    amrex::Real sx,
    amrex::Real sy,
    amrex::Real xlo,
    amrex::Real ylo,
    amrex::Real zsl,
    bool periodic,
    const amrex::Vector<amrex::MultiFab*>& fields,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& problo,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& dx,
    const LocalElevationReader& read,
    Interpolation interpolation = Interpolation::cubic);

} // namespace interp_to_mfab

#endif
