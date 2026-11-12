#include "velocity_stream.h"
#include "interp_to_mfab.h"
#include "AMReX_MultiFabUtil.H"
#include "gtest/gtest.h"
#include <memory>
#include <numeric>

namespace {

void compare_stream(bool periodic, bool subset, int ny = 6)
{
    const int nx = 8;
    amrex::Vector<amrex::Real> heights{2., 1.25, 0., -1., -3.};
    amrex::Vector<int> indices = subset ? amrex::Vector<int>{1, 2, 3}
                                        : amrex::Vector<int>{0, 1, 2, 3, 4};
    amrex::Vector<amrex::MultiFab*> actual_ptrs;
    std::vector<std::unique_ptr<amrex::MultiFab>> expected, actual;
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> problo, dx;
    std::size_t horizontal_cells = 0;
    for (int lev = 0; lev < 2; ++lev) {
        amrex::BoxArray boxes(
            amrex::Box(amrex::IntVect(0), amrex::IntVect(15, 11, 23)));
        boxes.maxSize(amrex::IntVect(8, 6, 8));
        amrex::DistributionMapping mapping(boxes);
        expected.push_back(
            std::make_unique<amrex::MultiFab>(boxes, mapping, 3, 2));
        actual.push_back(
            std::make_unique<amrex::MultiFab>(boxes, mapping, 3, 2));
        expected.back()->setVal(-999.);
        actual.back()->setVal(-999.);
        actual_ptrs.push_back(actual.back().get());
        // Shift the HOS origin; ghost cells cross both periodic seams.
        problo.push_back({.3, -.7, lev == 0 ? -4.25 : -4.});
        dx.push_back({.5 / (lev + 1), .5 / (lev + 1), .5});
        for (amrex::MFIter mfi(*actual.back()); mfi.isValid(); ++mfi) {
            const auto box = mfi.growntilebox();
            horizontal_cells +=
                static_cast<std::size_t>(box.length(0)) * box.length(1);
        }
    }
    const int plane = nx * ny;
    amrex::Gpu::DeviceVector<amrex::Real> u(plane * indices.size());
    amrex::Gpu::DeviceVector<amrex::Real> v(u.size()), w(u.size());
    auto* up = u.data();
    auto* vp = v.data();
    auto* wp = w.data();
    const int first = indices.front();
    amrex::ParallelFor(
        static_cast<int>(u.size()), [=] AMREX_GPU_DEVICE(int index) {
            const int h = first + index / plane;
            const int i = index % nx, j = (index % plane) / nx;
            const auto value = std::sin(.7 * i + .2 * j + h);
            up[index] = value;
            vp[index] = 2 * value - .3 * h;
            wp[index] = -.7 * value + .2 * h * h;
        });
    for (int lev = 0; lev < expected.size(); ++lev) {
        interp_to_mfab::interp_velocity_to_multifab(
            nx, ny, 1., ny == 1 ? 0. : 1., .3, -.7, periodic, indices, heights,
            u, v, w, *expected[lev], problo[lev], dx[lev]);
    }
    std::vector<int> calls;
    const auto memory = interp_to_mfab::stream_velocity_to_field(
        nx, ny, 1., ny == 1 ? 0. : 1., .3, -.7, periodic, indices, heights,
        actual_ptrs, problo, dx, [&](int height, auto& su, auto& sv, auto& sw) {
            calls.push_back(height);
            EXPECT_EQ(su.size(), plane);
            EXPECT_EQ(sv.size(), plane);
            EXPECT_EQ(sw.size(), plane);
            const int offset = (height - first) * plane;
            amrex::Gpu::copy(
                amrex::Gpu::deviceToDevice, u.begin() + offset,
                u.begin() + offset + plane, su.begin());
            amrex::Gpu::copy(
                amrex::Gpu::deviceToDevice, v.begin() + offset,
                v.begin() + offset + plane, sv.begin());
            amrex::Gpu::copy(
                amrex::Gpu::deviceToDevice, w.begin() + offset,
                w.begin() + offset + plane, sw.begin());
        });
    EXPECT_EQ(memory.plane_bytes, 3 * plane * sizeof(amrex::Real));
    EXPECT_EQ(
        memory.stencil_bytes, 24 * horizontal_cells * sizeof(amrex::Real));
    EXPECT_EQ(calls, std::vector<int>(indices.begin(), indices.end()));
    for (int lev = 0; lev < expected.size(); ++lev) {
        amrex::MultiFab::Subtract(*actual[lev], *expected[lev], 0, 0, 3, 2);
        for (int component = 0; component < 3; ++component) {
            EXPECT_EQ(actual[lev]->norm0(component, 2), 0.);
        }
    }
}

TEST(VelocityStream, PeriodicGhostCellsAndMultipleLevels)
{
    compare_stream(true, false);
}

TEST(VelocityStream, NonperiodicClampingAndGhostCells)
{
    compare_stream(false, false);
}

TEST(VelocityStream, SelectedHeightRangeAndExtrapolation)
{
    compare_stream(true, true);
}

TEST(VelocityStream, TwoDimensionalPeriodicAndNonperiodic)
{
    compare_stream(true, false, 1);
    compare_stream(false, true, 1);
}

TEST(VelocityStream, ScratchMemoryDoesNotGrowWithHeightCount)
{
    const amrex::Box box(amrex::IntVect(0), amrex::IntVect(3, 3, 3));
    amrex::BoxArray boxes(box);
    amrex::DistributionMapping mapping(boxes);
    amrex::MultiFab field(boxes, mapping, 3, 1);
    std::size_t bytes = 0;
    for (const int count : {10, 1000}) {
        amrex::Vector<amrex::Real> heights(count);
        amrex::Vector<int> indices(count);
        std::iota(indices.begin(), indices.end(), 0);
        for (int h = 0; h < count; ++h) {
            heights[h] = 10. - h;
        }
        int calls = 0;
        const auto memory = interp_to_mfab::stream_velocity_to_field(
            32, 32, 1., 1., 0., 0., true, indices, heights, {&field},
            {{{0., 0., 0.}}}, {{{1., 1., 1.}}},
            [&](int, auto& u, auto& v, auto& w) {
                ++calls;
                for (auto* values : {&u, &v, &w}) {
                    auto* ptr = values->data();
                    amrex::ParallelFor(
                        32 * 32, [=] AMREX_GPU_DEVICE(int n) { ptr[n] = 1.; });
                }
            });
        const auto total = memory.plane_bytes + memory.stencil_bytes;
        if (bytes != 0) {
            EXPECT_EQ(total, bytes);
        }
        bytes = total;
        EXPECT_LE(calls, 8);
        EXPECT_LT(total, 3 * 32 * 32 * count * sizeof(amrex::Real));
    }
}

TEST(VelocityStream, EmptyLocalMeshDoesNotReconstruct)
{
    int calls = 0;
    const auto memory = interp_to_mfab::stream_velocity_to_field(
        8, 8, 1., 1., 0., 0., true, {0, 1}, {1., 0.}, {}, {}, {},
        [&](int, auto&, auto&, auto&) { ++calls; });
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(memory.plane_bytes, 0);
    EXPECT_EQ(memory.stencil_bytes, 0);
}

} // namespace
