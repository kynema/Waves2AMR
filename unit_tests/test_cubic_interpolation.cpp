#include "interp_to_mfab.h"
#include "velocity_stream.h"
#include "gtest/gtest.h"
#include <cmath>
#include <memory>
#include <numeric>

namespace {

using interp_to_mfab::Interpolation;

AMREX_GPU_HOST_DEVICE
amrex::Real polynomial(amrex::Real x, amrex::Real y, amrex::Real z)
{
    return (1. + .2 * x - .03 * x * x + .004 * x * x * x) *
           (.8 + .1 * y + .02 * y * y - .003 * y * y * y) *
           (1. + .1 * z + .02 * z * z + .003 * z * z * z);
}

TEST(CubicInterpolation, StencilPolynomialNodesSmallGridsAndPeriodicSeams)
{
    for (int count : {1, 2, 3, 4, 8}) {
        for (amrex::Real coordinate : {-2., 0., .3, 1., 2.5, 6.7, 9.}) {
            const auto s =
                interp_to_mfab::horizontal_cubic(coordinate, count, 1., false);
            amrex::Real value = 0., sum = 0.;
            for (int a = 0; a < s.count; ++a) {
                EXPECT_GE(s.index[a], 0);
                EXPECT_LT(s.index[a], count);
                value += s.weight[a] *
                         std::pow(s.index[a] + 1., std::min(count - 1, 3));
                sum += s.weight[a];
            }
            const auto x = std::min(
                amrex::Real(count - 1), std::max(amrex::Real(0), coordinate));
            EXPECT_NEAR(value, std::pow(x + 1., std::min(count - 1, 3)), 1e-12);
            EXPECT_NEAR(sum, 1., 1e-14);
        }
    }
    for (amrex::Real x : {-.3, 0., .7, 7., 7.7, 8., 24.7}) {
        const auto a = interp_to_mfab::horizontal_cubic(x, 8, 1., true);
        const auto b = interp_to_mfab::horizontal_cubic(x + 24., 8, 1., true);
        for (int n = 0; n < 4; ++n) {
            EXPECT_EQ(a.index[n], b.index[n]);
            EXPECT_NEAR(a.weight[n], b.weight[n], 1e-14);
        }
    }
}

void polynomial_fields(
    int ny, int height_count, bool local, bool explicit_api = false)
{
    const int nx = 10;
    amrex::Vector<amrex::Real> heights{5., 3.2, 1.7, .4, -1.3, -4.};
    heights.resize(height_count);
    amrex::Vector<int> indices(height_count);
    std::iota(indices.begin(), indices.end(), 0);
    const amrex::GpuArray<amrex::Real, 3> plo{1., 1., heights.back() + .1};
    const amrex::GpuArray<amrex::Real, 3> dx{
        .31, .29, (heights.front() - heights.back() - .2) / 12};
    amrex::BoxArray boxes(
        amrex::Box(amrex::IntVect(0), amrex::IntVect(11, 9, 11)));
    boxes.maxSize(amrex::IntVect(6, 5, 4));
    amrex::DistributionMapping dm(boxes);
    amrex::MultiFab field(boxes, dm, 3, 0);
    field.setVal(-999.);
    const int plane = nx * ny;
    std::vector<amrex::Real> host(plane * height_count);
    for (int h = 0; h < height_count; ++h)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                host[h * plane + j * nx + i] =
                    polynomial(i, ny == 1 ? 0 : j, heights[h]);
    amrex::Gpu::DeviceVector<amrex::Real> volume(host.size());
    amrex::Gpu::copy(
        amrex::Gpu::hostToDevice, host.begin(), host.end(), volume.begin());
    interp_to_mfab::LocalVelocityPlaneReader reader;
    if (local)
        reader = [&](int h, const auto& xs, const auto& ys, auto& u, auto& v,
                     auto& w) {
            std::vector<amrex::Real> samples;
            for (int j : ys)
                for (int i : xs)
                    samples.push_back(host[h * plane + j * nx + i]);
            for (auto* values : {&u, &v, &w})
                amrex::Gpu::copy(
                    amrex::Gpu::hostToDevice, samples.begin(), samples.end(),
                    values->begin());
        };
    const auto memory = interp_to_mfab::stream_velocity_to_field(
        nx, ny, 1., 1., 0., 0., false, indices, heights, {&field}, {plo}, {dx},
        [&](int h, auto& u, auto& v, auto& w) {
            for (auto* values : {&u, &v, &w})
                amrex::Gpu::copy(
                    amrex::Gpu::deviceToDevice, volume.begin() + h * plane,
                    volume.begin() + (h + 1) * plane, values->begin());
        },
        reader);
    EXPECT_GT(memory.stencil_bytes, 0);
    if (explicit_api) {
        field.setVal(-999.);
        if (ny == 1)
            interp_to_mfab::interp_velocity_to_multifab2D(
                nx, 1., 0., false, indices, heights, volume, volume, field, plo,
                dx);
        else
            interp_to_mfab::interp_velocity_to_multifab3D(
                nx, ny, 1., 1., 0., 0., false, indices, heights, volume, volume,
                volume, field, plo, dx);
    }
    for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
        const auto values = field.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const auto x = plo[0] + (i + .5) * dx[0];
                const auto y = ny == 1 ? 0. : plo[1] + (j + .5) * dx[1];
                const auto z = plo[2] + (k + .5) * dx[2];
                const auto expected = polynomial(x, y, z);
                values(i, j, k, 0) -= expected;
                values(i, j, k, 1) -= ny == 1 ? 0. : expected;
                values(i, j, k, 2) -= expected;
            });
    }
    for (int c = 0; c < 3; ++c) EXPECT_LT(field.norm0(c), 2e-12);
}

TEST(CubicInterpolation, NonuniformVerticalAndTensorPolynomialReproduction)
{
    for (int ny : {1, 10})
        for (bool local : {false, true}) polynomial_fields(ny, 6, local);
    polynomial_fields(10, 4, true);
    polynomial_fields(1, 6, false, true);
    polynomial_fields(10, 6, false, true);
}

TEST(CubicInterpolation, ElevationPolynomialBoundariesAndGhosts)
{
    for (int ny : {1, 7}) {
        const int nx = 8;
        std::vector<amrex::Real> host(nx * ny);
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                host[j * nx + i] = polynomial(i, j, 0.);
        amrex::Gpu::DeviceVector<amrex::Real> eta(host.size());
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice, host.begin(), host.end(), eta.begin());
        amrex::BoxArray boxes(
            amrex::Box(amrex::IntVect(0), amrex::IntVect(9, 8, 2)));
        boxes.maxSize(5);
        amrex::DistributionMapping dm(boxes);
        amrex::MultiFab field(boxes, dm, 1, 1);
        const amrex::GpuArray<amrex::Real, 3> plo{-.5, -.5, 0.}, dx{.8, .8, .2};
        if (ny == 1)
            interp_to_mfab::interp_eta_to_levelset_multifab2D(
                nx, 1., 0., 2., false, eta, field, plo, dx);
        else
            interp_to_mfab::interp_eta_to_levelset_multifab3D(
                nx, ny, 1., 1., 0., 0., 2., false, eta, field, plo, dx);
        for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
            const auto values = field.array(mfi);
            amrex::ParallelFor(
                mfi.growntilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    const auto x = amrex::min(
                        amrex::Real(nx - 1),
                        amrex::max(amrex::Real(0), plo[0] + (i + .5) * dx[0]));
                    const auto y = amrex::min(
                        amrex::Real(ny - 1),
                        amrex::max(amrex::Real(0), plo[1] + (j + .5) * dx[1]));
                    values(i, j, k) -=
                        polynomial(x, y, 0.) + 2. - (plo[2] + (k + .5) * dx[2]);
                });
        }
        EXPECT_LT(field.norm0(0, 1), 2e-12);
    }
}

double fourier_error(int size, Interpolation method)
{
    const double length = 2 * M_PI;
    std::vector<amrex::Real> host(size * size);
    for (int j = 0; j < size; ++j)
        for (int i = 0; i < size; ++i)
            host[j * size + i] =
                std::cos(2 * length * i / size + length * j / size);
    amrex::Gpu::DeviceVector<amrex::Real> eta(host.size());
    amrex::Gpu::copy(
        amrex::Gpu::hostToDevice, host.begin(), host.end(), eta.begin());
    amrex::BoxArray boxes(
        amrex::Box(amrex::IntVect(0), amrex::IntVect(size - 1, size - 1, 0)));
    amrex::DistributionMapping dm(boxes);
    amrex::MultiFab field(boxes, dm, 1, 2);
    const amrex::GpuArray<amrex::Real, 3> plo{0., 0., 0.},
        dx{length / size, length / size, 1.};
    interp_to_mfab::interp_eta_to_levelset_multifab(
        size, size, length / size, length / size, 0., 0., 0., true, eta, field,
        plo, dx, method);
    for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
        const auto values = field.array(mfi);
        amrex::ParallelFor(
            mfi.growntilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                values(i, j, k) -=
                    std::cos(2 * (i + .5) * dx[0] + (j + .5) * dx[1]) -
                    (k + .5);
            });
    }
    return field.norm0(0, 2);
}

TEST(CubicInterpolation, PeriodicObliqueWaveFourthOrderConvergence)
{
    const auto coarse = fourier_error(16, Interpolation::cubic);
    const auto fine = fourier_error(32, Interpolation::cubic);
    const auto linear = fourier_error(32, Interpolation::linear);
    EXPECT_GT(coarse / fine, 13.);
    EXPECT_LT(fine, linear * .05);
}

TEST(CubicInterpolation, SmallHeightSetsReduceOrderAndPreserveExtrapolation)
{
    for (int count : {2, 3, 4}) {
        amrex::Vector<amrex::Real> heights{3., 1., -2., -4.};
        heights.resize(count);
        amrex::Vector<int> indices(count);
        std::iota(indices.begin(), indices.end(), 0);
        const amrex::Box box(amrex::IntVect(0), amrex::IntVect(0, 0, 9));
        amrex::BoxArray boxes(box);
        amrex::DistributionMapping dm(boxes);
        amrex::MultiFab field(boxes, dm, 3, 0);
        const amrex::GpuArray<amrex::Real, 3> plo{0., 0., -5.5}, dx{1., 1., 1.};
        interp_to_mfab::stream_velocity_to_field(
            4, 1, 1., 0., 0., 0., true, indices, heights, {&field}, {plo}, {dx},
            [&](int h, auto& u, auto& v, auto& w) {
                const double value = std::pow(heights[h], count - 1);
                for (auto* component : {&u, &v, &w}) {
                    auto* values = component->data();
                    amrex::ParallelFor(
                        4, [=] AMREX_GPU_DEVICE(int n) { values[n] = value; });
                }
            });
        std::vector<amrex::Real> expected(10);
        for (int k = 0; k < 10; ++k) {
            const double z = -5. + k;
            if (z - .5 > heights.front())
                expected[k] = 0.;
            else if (z > heights.front() || z < heights.back()) {
                int upper = z > heights.front() ? 0 : count - 2;
                const double t = (z - heights[upper]) /
                                 (heights[upper + 1] - heights[upper]);
                expected[k] = (1 - t) * std::pow(heights[upper], count - 1) +
                              t * std::pow(heights[upper + 1], count - 1);
            } else
                expected[k] = std::pow(z, count - 1);
        }
        amrex::Gpu::DeviceVector<amrex::Real> reference(expected.size());
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice, expected.begin(), expected.end(),
            reference.begin());
        const auto* ep = reference.data();
        for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
            const auto values = field.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    values(i, j, k, 0) -= ep[k];
                    values(i, j, k, 2) -= ep[k];
                });
        }
        for (int c = 0; c < 3; ++c) EXPECT_LT(field.norm0(c), 1e-12);
    }
}

double vertical_error(double scale, Interpolation method)
{
    amrex::Vector<amrex::Real> heights{1., .6, .1, -.5, -1.};
    for (auto& h : heights) h *= scale;
    amrex::BoxArray boxes(amrex::Box(amrex::IntVect(0), amrex::IntVect(0)));
    amrex::DistributionMapping dm(boxes);
    amrex::MultiFab field(boxes, dm, 3, 0);
    const double z = .2 * scale;
    interp_to_mfab::stream_velocity_to_field(
        4, 1, 1., 0., 0., 0., true, {0, 1, 2, 3, 4}, heights, {&field},
        {{{0., 0., z - .005}}}, {{{1., 1., .01}}},
        [&](int h, auto& u, auto& v, auto& w) {
            const auto value = std::exp(heights[h]);
            for (auto* component : {&u, &v, &w}) {
                auto* values = component->data();
                amrex::ParallelFor(
                    4, [=] AMREX_GPU_DEVICE(int n) { values[n] = value; });
            }
        },
        {}, method);
    return std::abs(field.max(0) - std::exp(z));
}

TEST(CubicInterpolation, NonuniformVerticalFourthOrderConvergence)
{
    const auto coarse = vertical_error(1., Interpolation::cubic);
    const auto fine = vertical_error(.5, Interpolation::cubic);
    EXPECT_GT(coarse / fine, 13.);
    EXPECT_LT(fine, vertical_error(.5, Interpolation::linear) * .02);
}
} // namespace
