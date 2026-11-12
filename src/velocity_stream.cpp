#include "velocity_stream.h"
#include "interp_to_mfab.h"
#include "AMReX_FArrayBox.H"
#include <memory>
#include <algorithm>

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

struct LocalGrid
{
    std::vector<int> xs, ys;
    amrex::Gpu::DeviceVector<int> xmap, ymap;
};

LocalGrid local_grid(
    int nx,
    int ny,
    amrex::Real sx,
    amrex::Real sy,
    amrex::Real xlo,
    amrex::Real ylo,
    bool periodic,
    const amrex::Vector<amrex::MultiFab*>& fields,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& problo,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& dx,
    interp_to_mfab::Interpolation interpolation)
{
    LocalGrid grid;
    std::vector<int> xm(nx, -1), ym(ny, -1);
    for (int lev = 0; lev < fields.size(); ++lev) {
        for (amrex::MFIter mfi(*fields[lev]); mfi.isValid(); ++mfi) {
            const auto cells = mfi.growntilebox();
            for (int i = cells.smallEnd(0); i <= cells.bigEnd(0); ++i) {
                if (interpolation == interp_to_mfab::Interpolation::cubic) {
                    const auto s = interp_to_mfab::horizontal_cubic(
                        problo[lev][0] + (i + .5) * dx[lev][0] - xlo, nx, sx,
                        periodic);
                    for (int a = 0; a < s.count; ++a) xm[s.index[a]] = 0;
                    continue;
                }
                const auto s = stencil(
                    i, 0, nx, ny, sx, sy, xlo, ylo, periodic, problo[lev],
                    dx[lev]);
                AMREX_ALWAYS_ASSERT(
                    s.i0 >= 0 && s.i1 >= 0 && s.i0 < nx && s.i1 < nx);
                xm[s.i0] = xm[s.i1] = 0;
            }
            for (int j = cells.smallEnd(1); j <= cells.bigEnd(1); ++j) {
                if (interpolation == interp_to_mfab::Interpolation::cubic) {
                    const auto s = interp_to_mfab::horizontal_cubic(
                        problo[lev][1] + (j + .5) * dx[lev][1] - ylo, ny, sy,
                        periodic);
                    for (int a = 0; a < s.count; ++a) ym[s.index[a]] = 0;
                    continue;
                }
                const auto s = stencil(
                    0, j, nx, ny, sx, sy, xlo, ylo, periodic, problo[lev],
                    dx[lev]);
                AMREX_ALWAYS_ASSERT(
                    s.j0 >= 0 && s.j1 >= 0 && s.j0 < ny && s.j1 < ny);
                ym[s.j0] = ym[s.j1] = 0;
            }
        }
    }
    for (int i = 0; i < nx; ++i) {
        if (xm[i] == 0) {
            xm[i] = grid.xs.size();
            grid.xs.push_back(i);
        }
    }
    for (int j = 0; j < ny; ++j) {
        if (ym[j] == 0) {
            ym[j] = grid.ys.size();
            grid.ys.push_back(j);
        }
    }
    grid.xmap.resize(nx);
    grid.ymap.resize(ny);
    amrex::Gpu::copy(
        amrex::Gpu::hostToDevice, xm.begin(), xm.end(), grid.xmap.begin());
    amrex::Gpu::copy(
        amrex::Gpu::hostToDevice, ym.begin(), ym.end(), grid.ymap.begin());
    return grid;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real horizontal_value(
    const amrex::Real* values,
    const interp_to_mfab::PolynomialStencil& x,
    const interp_to_mfab::PolynomialStencil& y,
    int width,
    const int* xm,
    const int* ym)
{
    amrex::Real result = 0.;
    for (int b = 0; b < y.count; ++b) {
        amrex::Real row = 0.;
        const int j = ym ? ym[y.index[b]] : y.index[b];
        for (int a = 0; a < x.count; ++a) {
            const int i = xm ? xm[x.index[a]] : x.index[a];
            row += x.weight[a] * values[i + j * width];
        }
        result += y.weight[b] * row;
    }
    return result;
}

interp_to_mfab::VelocityStreamMemory stream_cubic(
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
    const interp_to_mfab::VelocityPlaneReader& read_plane,
    const interp_to_mfab::LocalVelocityPlaneReader& read_local_plane)
{
    using namespace interp_to_mfab;
    const int first = indices.front(), last = indices.back();
    const auto top_height = heights[first], bottom_height = heights[last];
    const int depth_count = amrex::min(4, last - first + 1);
    VelocityStreamMemory memory;
    std::vector<LocalPatch> patches;
    for (int lev = 0; lev < fields.size(); ++lev) {
        AMREX_ALWAYS_ASSERT(fields[lev] && fields[lev]->nComp() >= 3);
        for (amrex::MFIter mfi(*fields[lev]); mfi.isValid(); ++mfi) {
            const auto cells = mfi.growntilebox();
            const auto plo = problo[lev], spacing = dx[lev];
            const int top = upper_height(
                plo[2] + (cells.bigEnd(2) + .5) * spacing[2], heights.data(),
                first, last);
            const int bottom = upper_height(
                plo[2] + (cells.smallEnd(2) + .5) * spacing[2], heights.data(),
                first, last);
            auto horizontal = cells;
            horizontal.setSmall(2, 0);
            horizontal.setBig(2, 0);
            memory.stencil_bytes +=
                horizontal.numPts() * 3 * depth_count * sizeof(amrex::Real);
            patches.push_back(
                {cells, fields[lev]->array(mfi), plo, spacing,
                 vertical_stencil_start(top, first, last),
                 vertical_stencil_start(bottom, first, last) + depth_count - 1,
                 std::make_unique<amrex::FArrayBox>(
                     horizontal, 3 * depth_count)});
        }
    }
    if (patches.empty()) return memory;
    const bool local = static_cast<bool>(read_local_plane);
    LocalGrid grid;
    if (local)
        grid = local_grid(
            nx, ny, sx, sy, xlo, ylo, periodic, fields, problo, dx,
            Interpolation::cubic);
    const auto count =
        local ? grid.xs.size() * grid.ys.size() : std::size_t(nx) * ny;
    const int width = local ? grid.xs.size() : nx;
    const auto* xm = local ? grid.xmap.data() : nullptr;
    const auto* ym = local ? grid.ymap.data() : nullptr;
    amrex::Gpu::DeviceVector<amrex::Real> u(count), v(count), w(count);
    memory.plane_bytes = 3 * count * sizeof(amrex::Real);
    for (int h = first; h <= last; ++h) {
        bool needed = false;
        for (const auto& patch : patches)
            needed = needed || (h >= patch.first && h <= patch.last);
        if (!needed) continue;
        if (local)
            read_local_plane(h, grid.xs, grid.ys, u, v, w);
        else
            read_plane(h, u, v, w);
        AMREX_ALWAYS_ASSERT(
            u.size() == count && v.size() == count && w.size() == count);
        const auto* up = u.data();
        const auto* vp = v.data();
        const auto* wp = w.data();
        for (const auto& patch : patches) {
            if (h < patch.first || h > patch.last) continue;
            const auto samples = patch.samples->array();
            const auto plo = patch.plo, spacing = patch.dx;
            const int offset = 3 * (h % depth_count);
            amrex::ParallelFor(
                patch.samples->box(), [=] AMREX_GPU_DEVICE(int i, int j, int) {
                    const auto x = horizontal_cubic(
                        plo[0] + (i + .5) * spacing[0] - xlo, nx, sx, periodic);
                    const auto y = horizontal_cubic(
                        plo[1] + (j + .5) * spacing[1] - ylo, ny, sy, periodic);
                    samples(i, j, 0, offset) =
                        horizontal_value(up, x, y, width, xm, ym);
                    samples(i, j, 0, offset + 1) =
                        ny == 1 ? 0.
                                : horizontal_value(vp, x, y, width, xm, ym);
                    samples(i, j, 0, offset + 2) =
                        horizontal_value(wp, x, y, width, xm, ym);
                });
            if (h < patch.first + depth_count - 1) continue;
            const int start = h - depth_count + 1;
            amrex::GpuArray<amrex::Real, 4> nodes{};
            for (int n = 0; n < depth_count; ++n) nodes[n] = heights[start + n];
            const auto output = patch.output;
            // Several boundary intervals can share the same one-sided stencil.
            for (int upper = amrex::max(first, start - 1);
                 upper <= amrex::min(last - 1, h); ++upper) {
                if (vertical_stencil_start(upper, first, last) != start)
                    continue;
                const auto zhi = heights[upper], zlo = heights[upper + 1];
                amrex::ParallelFor(
                    patch.cells, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        const auto z = plo[2] + (k + .5) * spacing[2];
                        if ((upper != first && z >= zhi) ||
                            (upper != last - 1 && z < zlo))
                            return;
                        if (z - .5 * spacing[2] > zhi) {
                            for (int c = 0; c < 3; ++c) output(i, j, k, c) = 0.;
                            return;
                        }
                        // Keep the legacy linear extrapolation outside the
                        // height range.
                        const bool outside =
                            z > top_height || z < bottom_height;
                        const int begin = outside ? upper : start;
                        const int nweights = outside ? 2 : depth_count;
                        amrex::Real zs[4];
                        for (int n = 0; n < nweights; ++n)
                            zs[n] = outside ? (n == 0 ? zhi : zlo) : nodes[n];
                        const auto weights =
                            lagrange_stencil(z, zs, begin, nweights);
                        for (int c = 0; c < 3; ++c) {
                            amrex::Real value = 0.;
                            for (int n = 0; n < nweights; ++n)
                                value +=
                                    weights.weight[n] *
                                    samples(
                                        i, j, 0,
                                        3 * (weights.index[n] % depth_count) +
                                            c);
                            output(i, j, k, c) = value;
                        }
                    });
            }
        }
        amrex::Gpu::streamSynchronize();
    }
    return memory;
}

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
    const VelocityPlaneReader& read_plane,
    const LocalVelocityPlaneReader& read_local_plane,
    Interpolation interpolation)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        nx > 1 && ny >= 1 && sx > 0 && (ny == 1 || sy > 0) &&
            indices.size() >= 2 && fields.size() == problo.size() &&
            fields.size() == dx.size() &&
            (static_cast<bool>(read_plane) ||
             static_cast<bool>(read_local_plane)),
        "Waves2AMR: invalid velocity streaming inputs");
    for (int n = 0; n < indices.size(); ++n) {
        AMREX_ALWAYS_ASSERT(indices[n] >= 0 && indices[n] < heights.size());
        AMREX_ALWAYS_ASSERT(n == 0 || indices[n] == indices[n - 1] + 1);
    }
    for (int n = 1; n < heights.size(); ++n) {
        AMREX_ALWAYS_ASSERT(heights[n] < heights[n - 1]);
    }
    if (interpolation == Interpolation::cubic)
        return stream_cubic(
            nx, ny, sx, sy, xlo, ylo, periodic, indices, heights, fields,
            problo, dx, read_plane, read_local_plane);
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
    LocalGrid grid;
    if (read_local_plane)
        grid = local_grid(
            nx, ny, sx, sy, xlo, ylo, periodic, fields, problo, dx,
            interpolation);
    const bool local = static_cast<bool>(read_local_plane);
    const auto count = local ? grid.xs.size() * grid.ys.size()
                             : static_cast<std::size_t>(nx) * ny;
    const auto* xm = grid.xmap.data();
    const auto* ym = grid.ymap.data();
    const int width = local ? grid.xs.size() : nx;
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
        if (local)
            read_local_plane(h, grid.xs, grid.ys, u, v, w);
        else
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
                    const int i0 = local ? xm[s.i0] : s.i0;
                    const int i1 = local ? xm[s.i1] : s.i1;
                    const int j0 = local ? ym[s.j0] : s.j0;
                    const int j1 = local ? ym[s.j1] : s.j1;
                    const int corners[4] = {
                        i0 + j0 * width, i1 + j0 * width, i0 + j1 * width,
                        i1 + j1 * width};
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

void interp_to_mfab::local_eta_to_levelset_field(
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
    Interpolation interpolation)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        nx > 1 && ny >= 1 && sx > 0 && (ny == 1 || sy > 0) &&
            fields.size() == problo.size() && fields.size() == dx.size() &&
            static_cast<bool>(read),
        "Waves2AMR: invalid local elevation inputs");
    for (auto* field : fields)
        AMREX_ALWAYS_ASSERT(field && field->nComp() >= 1);
    auto grid = local_grid(
        nx, ny, sx, sy, xlo, ylo, periodic, fields, problo, dx, interpolation);
    if (grid.xs.empty() || grid.ys.empty()) return;
    amrex::Gpu::DeviceVector<amrex::Real> eta(grid.xs.size() * grid.ys.size());
    read(grid.xs, grid.ys, eta);
    AMREX_ALWAYS_ASSERT(eta.size() == grid.xs.size() * grid.ys.size());
    const auto* values = eta.data();
    const auto* xm = grid.xmap.data();
    const auto* ym = grid.ymap.data();
    const int width = grid.xs.size();
    for (int lev = 0; lev < fields.size(); ++lev) {
        const auto plo = problo[lev], spacing = dx[lev];
        for (amrex::MFIter mfi(*fields[lev]); mfi.isValid(); ++mfi) {
            const auto output = fields[lev]->array(mfi);
            amrex::ParallelFor(
                mfi.growntilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if (interpolation == Interpolation::cubic) {
                        const auto x = horizontal_cubic(
                            plo[0] + (i + .5) * spacing[0] - xlo, nx, sx,
                            periodic);
                        const auto y = horizontal_cubic(
                            plo[1] + (j + .5) * spacing[1] - ylo, ny, sy,
                            periodic);
                        output(i, j, k) =
                            horizontal_value(values, x, y, width, xm, ym) +
                            zsl - (plo[2] + (k + .5) * spacing[2]);
                        return;
                    }
                    const auto s = stencil(
                        i, j, nx, ny, sx, sy, xlo, ylo, periodic, plo, spacing);
                    const auto a = values[xm[s.i0] + ym[s.j0] * width];
                    const auto b = values[xm[s.i1] + ym[s.j0] * width];
                    const auto c = values[xm[s.i0] + ym[s.j1] * width];
                    const auto d = values[xm[s.i1] + ym[s.j1] * width];
                    const auto height =
                        ny == 1
                            ? linear_interp1D(a, b, s.x, s.x0, s.x1)
                            : linear_interp2D(
                                  a, b, c, d, s.x, s.y, s.x0, s.y0, s.x1, s.y1);
                    output(i, j, k) =
                        height + zsl - (plo[2] + (k + .5) * spacing[2]);
                });
        }
    }
    amrex::Gpu::streamSynchronize();
}
