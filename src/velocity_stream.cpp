#include "velocity_stream.h"
#include "interp_to_mfab.h"
#include "AMReX_FArrayBox.H"
#include <memory>

namespace {

struct Stencil
{
    int i0, i1, j0, j1;
    amrex::Real x, y, x0, x1, y0, y1;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Stencil stencil(
    int i,
    int j,
    int nx,
    int ny,
    amrex::Real sx,
    amrex::Real sy,
    amrex::Real xlo,
    amrex::Real ylo,
    bool periodic,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx)
{
    Stencil s;
    if (ny == 1) {
        sy = 1.;
    }
    s.x = plo[0] + (i + 0.5) * dx[0] - xlo;
    s.y = ny == 1 ? 0. : plo[1] + (j + 0.5) * dx[1] - ylo;
    const auto lx = nx * sx, ly = ny * sy;
    if (periodic) {
        s.x = s.x > lx ? s.x - lx : s.x;
        s.x = s.x < 0 ? s.x + lx : s.x;
        s.y = s.y > ly ? s.y - ly : s.y;
        s.y = s.y < 0 ? s.y + ly : s.y;
    } else {
        s.x = amrex::min(lx, amrex::max(0., s.x));
        s.y = amrex::min(ly, amrex::max(0., s.y));
    }
    s.i0 = s.x / sx;
    s.j0 = s.y / sy;
    s.x0 = sx * s.i0;
    s.y0 = sy * s.j0;
    // Match the existing interpolator, including its endpoint conventions.
    while (s.i0 < nx - 2 && s.x0 - sx < s.x) {
        s.x0 = sx * ++s.i0;
    }
    while ((s.i0 > 0 && s.x0 > s.x) || s.i0 >= nx) {
        s.x0 = sx * --s.i0;
    }
    while (s.j0 < ny - 2 && s.y0 - sy < s.y) {
        s.y0 = sy * ++s.j0;
    }
    while ((s.j0 > 0 && s.y0 > s.y) || s.j0 >= ny) {
        s.y0 = sy * --s.j0;
    }
    s.i1 = s.i0 + 1;
    s.j1 = s.j0 + 1;
    s.x1 = sx * s.i1;
    s.y1 = sy * s.j1;
    if (periodic) {
        s.i1 = s.i1 >= nx ? s.i1 - nx : s.i1;
        s.j1 = s.j1 >= ny ? s.j1 - ny : s.j1;
    } else {
        s.i0 += s.i1 >= nx ? -1 : 0;
        s.i1 += s.i1 >= nx ? -1 : 0;
        s.j0 += s.j1 >= ny ? -1 : 0;
        s.j1 += s.j1 >= ny ? -1 : 0;
    }
    if (ny == 1) {
        s.j0 = s.j1 = 0;
    }
    return s;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
upper_height(amrex::Real z, const amrex::Real* heights, int first, int last)
{
    int upper = first;
    while (upper < last - 1 && heights[upper + 1] > z) {
        ++upper;
    }
    return upper;
}

struct LocalPatch
{
    amrex::Box cells;
    amrex::Array4<amrex::Real> output;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo, dx;
    int first, last;
    std::unique_ptr<amrex::FArrayBox> samples;
};

} // namespace

interp_to_mfab::VelocityStreamMemory interp_to_mfab::stream_velocity_to_field(
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
    const VelocityPlaneReader& read_plane)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        nx > 1 && ny >= 1 && sx > 0 && (ny == 1 || sy > 0) &&
            indices.size() >= 2 && fields.size() == problo.size() &&
            fields.size() == dx.size() && static_cast<bool>(read_plane),
        "Waves2AMR: invalid velocity streaming inputs");
    for (int n = 0; n < indices.size(); ++n) {
        AMREX_ALWAYS_ASSERT(indices[n] >= 0 && indices[n] < heights.size());
        AMREX_ALWAYS_ASSERT(n == 0 || indices[n] == indices[n - 1] + 1);
    }
    for (int n = 1; n < heights.size(); ++n) {
        AMREX_ALWAYS_ASSERT(heights[n] < heights[n - 1]);
    }
    VelocityStreamMemory memory;
    std::vector<LocalPatch> patches;
    const int first = indices.front(), last = indices.back();
    for (int lev = 0; lev < fields.size(); ++lev) {
        AMREX_ALWAYS_ASSERT(
            fields[lev] != nullptr && fields[lev]->nComp() >= 3);
        for (amrex::MFIter mfi(*fields[lev]); mfi.isValid(); ++mfi) {
            const auto cells = mfi.growntilebox();
            const auto plo = problo[lev], spacing = dx[lev];
            const auto top = plo[2] + (cells.bigEnd(2) + 0.5) * spacing[2];
            const auto bottom = plo[2] + (cells.smallEnd(2) + 0.5) * spacing[2];
            const int start = upper_height(top, heights.data(), first, last);
            const int stop =
                upper_height(bottom, heights.data(), first, last) + 1;
            auto horizontal = cells;
            horizontal.setSmall(2, 0);
            horizontal.setBig(2, 0);
            memory.stencil_bytes +=
                horizontal.numPts() * 24 * sizeof(amrex::Real);
            patches.push_back(
                {cells, fields[lev]->array(mfi), plo, spacing, start, stop,
                 std::make_unique<amrex::FArrayBox>(horizontal, 24)});
        }
    }
    if (patches.empty()) {
        return memory;
    }
    const auto count = static_cast<std::size_t>(nx) * ny;
    amrex::Gpu::DeviceVector<amrex::Real> u(count), v(count), w(count);
    memory.plane_bytes = 3 * count * sizeof(amrex::Real);
    for (int h = first; h <= last; ++h) {
        bool needed = false;
        for (const auto& patch : patches) {
            needed = needed || (h >= patch.first && h <= patch.last);
        }
        if (!needed) {
            continue;
        }
        read_plane(h, u, v, w);
        AMREX_ALWAYS_ASSERT(
            u.size() == count && v.size() == count && w.size() == count);
        const auto* up = u.data();
        const auto* vp = v.data();
        const auto* wp = w.data();
        const int offset = 12 * (h % 2);
        for (const auto& patch : patches) {
            if (h < patch.first || h > patch.last) {
                continue;
            }
            const auto samples = patch.samples->array();
            const auto plo = patch.plo, spacing = patch.dx;
            amrex::ParallelFor(
                patch.samples->box(), [=] AMREX_GPU_DEVICE(int i, int j, int) {
                    const auto s = stencil(
                        i, j, nx, ny, sx, sy, xlo, ylo, periodic, plo, spacing);
                    const int corners[4] = {
                        s.i0 + s.j0 * nx, s.i1 + s.j0 * nx, s.i0 + s.j1 * nx,
                        s.i1 + s.j1 * nx};
                    for (int c = 0; c < 4; ++c) {
                        samples(i, j, 0, offset + c) = up[corners[c]];
                        samples(i, j, 0, offset + 4 + c) = vp[corners[c]];
                        samples(i, j, 0, offset + 8 + c) = wp[corners[c]];
                    }
                });
            if (h == patch.first) {
                continue;
            }
            const auto output = patch.output;
            const auto z0 = heights[h], z1 = heights[h - 1];
            const int previous = 12 * ((h - 1) % 2);
            amrex::ParallelFor(
                patch.cells, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    const auto z = plo[2] + (k + 0.5) * spacing[2];
                    // Exact-height cells belong to the pair above that height.
                    if ((h != last && z < z0) || (h - 1 != first && z >= z1)) {
                        return;
                    }
                    if (z - 0.5 * spacing[2] > z1) {
                        for (int n = 0; n < 3; ++n) {
                            output(i, j, k, n) = 0.;
                        }
                        return;
                    }
                    const auto s = stencil(
                        i, j, nx, ny, sx, sy, xlo, ylo, periodic, plo, spacing);
                    for (int n = 0; n < 3; ++n) {
                        const int a = offset + 4 * n, b = previous + 4 * n;
                        if (ny == 1) {
                            output(i, j, k, n) =
                                n == 1 ? 0.
                                       : linear_interp2D(
                                             samples(i, j, 0, a),
                                             samples(i, j, 0, a + 1),
                                             samples(i, j, 0, b),
                                             samples(i, j, 0, b + 1), s.x, z,
                                             s.x0, z0, s.x1, z1);
                            continue;
                        }
                        output(i, j, k, n) = linear_interp(
                            samples(i, j, 0, a), samples(i, j, 0, a + 1),
                            samples(i, j, 0, a + 2), samples(i, j, 0, b),
                            samples(i, j, 0, a + 3), samples(i, j, 0, b + 1),
                            samples(i, j, 0, b + 2), samples(i, j, 0, b + 3),
                            s.x, s.y, z, s.x0, s.y0, z0, s.x1, s.y1, z1);
                    }
                });
        }
        // The next host FFT/copy may overwrite the single global plane.
        amrex::Gpu::streamSynchronize();
    }
    return memory;
}
