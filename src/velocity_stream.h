#ifndef VELOCITY_STREAM_H
#define VELOCITY_STREAM_H

#include "AMReX_MultiFab.H"
#include <cstddef>
#include <functional>

namespace interp_to_mfab {

using VelocityPlaneReader = std::function<void(
    int,
    amrex::Gpu::DeviceVector<amrex::Real>&,
    amrex::Gpu::DeviceVector<amrex::Real>&,
    amrex::Gpu::DeviceVector<amrex::Real>&)>;

struct VelocityStreamMemory
{
    std::size_t plane_bytes{0};
    std::size_t stencil_bytes{0};
};

// The reader fills one full FFT plane per component at the requested height
// index. Only two local horizontal interpolation stencils are retained.
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
    const VelocityPlaneReader& read_plane);

} // namespace interp_to_mfab

#endif
