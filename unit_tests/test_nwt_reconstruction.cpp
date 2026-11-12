#include "nwt_reconstruction.h"
#include "modes_hosgrid.h"
#include "interp_to_mfab.h"
#include "gtest/gtest.h"
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace {
using modes_hosgrid::NwtReconstruction;

TEST(NwtReconstruction, EndpointPaddingAndCaps)
{
    EXPECT_EQ(
        modes_hosgrid::nwt_fft_dimensions(5, 9, 4., 8., .26, .26),
        (std::array<int, 2>{17, 33}));
    EXPECT_EQ(
        modes_hosgrid::nwt_fft_dimensions(5, 9, 4., 8., .1, .1, 1),
        (std::array<int, 2>{9, 17}));
    EXPECT_EQ(
        modes_hosgrid::nwt_fft_dimensions(1, 9, 0., 8., 0., .1, 0),
        (std::array<int, 2>{1, 9}));
    EXPECT_THROW(
        modes_hosgrid::nwt_fft_dimensions(2, 9, 4., 8., .1, .1),
        std::invalid_argument);
}

TEST(NwtReconstruction, AllParitiesEndpointsAndPaddedSeries)
{
    for (int sy : {1, 4, 5})
        for (int sx : {4, 5}) {
            std::vector<double> modes(sy * sx);
            for (int i = 0; i < sy * sx; ++i) modes[i] = std::sin(i + .3);
            for (int factor : {1, 2, 4}) {
                const int ny = sy == 1 ? 1 : factor * (sy - 1) + 1,
                          nx = factor * (sx - 1) + 1;
                for (const std::string method : {"full", "pruned", "auto"}) {
                    NwtReconstruction fft(sy, sx, ny, nx, {method});
                    const std::vector<int> xs{0, 1, nx - 1},
                        ys = ny == 1 ? std::vector<int>{0}
                                     : std::vector<int>{0, 1, ny - 1};
                    for (bool cy : {false, true})
                        for (bool cx : {false, true}) {
                            const auto actual =
                                fft.evaluate(modes, cy, cx, xs, ys);
                            for (std::size_t y = 0; y < ys.size(); ++y)
                                for (std::size_t x = 0; x < xs.size(); ++x) {
                                    double expected = 0.;
                                    for (int j = 0; j < sy; ++j)
                                        for (int i = 0; i < sx; ++i) {
                                            if ((!cx &&
                                                 (i == 0 || i == sx - 1)) ||
                                                (!cy &&
                                                 (j == 0 || j == sy - 1)))
                                                continue;
                                            const double ax =
                                                M_PI * i * xs[x] / (nx - 1);
                                            const double ay =
                                                ny == 1 ? 0.
                                                        : M_PI * j * ys[y] /
                                                              (ny - 1);
                                            expected += modes[j * sx + i] *
                                                        (cx ? std::cos(ax)
                                                            : std::sin(ax)) *
                                                        (cy ? std::cos(ay)
                                                            : std::sin(ay));
                                        }
                                    EXPECT_NEAR(
                                        actual[y * xs.size() + x], expected,
                                        3e-13);
                                }
                        }
                    EXPECT_EQ(fft.stats().selections, 1);
                }
            }
        }
}

TEST(NwtReconstruction, VelocityAndWavemakerMatchLegacy)
{
    for (int sy : {1, 5}) {
        const int sx = 7, count = sy * sx;
        std::vector<double> mx(count), my(count), mz(count), add(sy * 3);
        for (int i = 0; i < count; ++i) {
            mx[i] = .1 * std::sin(i + .2);
            my[i] = .1 * std::cos(i + .4);
            mz[i] = .1 * std::sin(i + .7);
        }
        for (std::size_t i = 0; i < add.size(); ++i)
            add[i] = .1 * std::sin(i + .9);
        std::vector<double> scratch(count), out(count), a(sy), b(sy), c(sy);
        std::vector<fftw_plan> plans;
        modes_hosgrid::plan_ifftw_nwt(
            sy, sx, plans, scratch.data(), out.data(),
            modes_hosgrid::planner_flags::estimate);
        for (bool include_add : {false, true})
            for (double z : {-.8, 0., .3}) {
                const auto& additional =
                    include_add ? add : std::vector<double>{};
                amrex::Gpu::DeviceVector<amrex::Real> u(count), v(count),
                    w(count);
                std::vector<double> xm(count), ym(count), zm(count);
                modes_hosgrid::populate_hos_vel(
                    sy, sx, include_add ? 3 : 0, 8., 4., z, 0., 1., 2., mx, my,
                    mz, additional, plans, xm.data(), ym.data(), zm.data(),
                    a.data(), b.data(), c.data(), u, v, w);
                std::array<std::vector<double>, 3> reference;
                for (int component = 0; component < 3; ++component) {
                    const auto& source = component == 0   ? u
                                         : component == 1 ? v
                                                          : w;
                    reference[component].resize(count);
                    amrex::Gpu::copy(
                        amrex::Gpu::deviceToHost, source.begin(), source.end(),
                        reference[component].begin());
                }
                for (int factor : {1, 2})
                    for (const std::string method :
                         {"full", "pruned", "auto"}) {
                        const int ny = sy == 1 ? 1 : factor * (sy - 1) + 1,
                                  nx = factor * (sx - 1) + 1;
                        std::vector<int> xs(sx), ys(sy);
                        for (int i = 0; i < sx; ++i) xs[i] = i * factor;
                        for (int j = 0; j < sy; ++j) ys[j] = j * factor;
                        NwtReconstruction fft(sy, sx, ny, nx, {method});
                        const auto actual = fft.velocity(
                            8., 4., z, 0., 1., 2., mx, my, mz, additional, xs,
                            ys);
                        for (int component = 0; component < 3; ++component)
                            for (int i = 0; i < count; ++i)
                                EXPECT_NEAR(
                                    actual[component][i],
                                    reference[component][i], 2e-13);
                    }
            }
        for (auto plan : plans) fftw_destroy_plan(plan);
    }
}

TEST(NwtReconstruction, BudgetIndicesAndPlanner)
{
    std::vector<double> modes(5 * 9, .1);
    NwtReconstruction fft(5, 9, 257, 513, {"auto", 128 * 1024, FFTW_MEASURE});
    const auto first = fft.evaluate(modes, true, true, {0, 3, 512}, {0, 256});
    EXPECT_NE(fft.stats().method, "full");
    EXPECT_LE(fft.stats().scratch_bytes, 128 * 1024);
    EXPECT_EQ(first, fft.evaluate(modes, true, true, {0, 3, 512}, {0, 256}));
    EXPECT_EQ(fft.stats().selections, 1);
    EXPECT_THROW(
        fft.evaluate(modes, true, true, {0}, {0}, 128 * 1024),
        std::runtime_error);
    EXPECT_THROW(
        fft.evaluate(modes, true, true, {513}, {0}), std::invalid_argument);
    EXPECT_THROW(
        fft.evaluate(modes, true, true, {1, 0}, {0}), std::invalid_argument);
    EXPECT_TRUE(fft.evaluate(modes, true, true, {}, {}).empty());
    EXPECT_THROW(fft.evaluate({}, true, true, {0}, {0}), std::invalid_argument);
    EXPECT_EQ(modes, std::vector<double>(5 * 9, .1));
}

TEST(NwtReconstruction, WavemakerAtNewPaddedNodes)
{
    for (int sy : {1, 5}) {
        const int sx = 7, ny = sy == 1 ? 1 : 17, nx = 25, count = 3;
        std::vector<double> zero(sy * sx), add(sy * count);
        for (std::size_t n = 0; n < add.size(); ++n)
            add[n] = .2 * std::cos(n + .3);
        const std::vector<int> xs{0, 1, 7, 24};
        const std::vector<int> ys =
            sy == 1 ? std::vector<int>{0} : std::vector<int>{0, 1, 5, 16};
        NwtReconstruction fft(sy, sx, ny, nx, {"pruned"});
        const double ly = sy == 1 ? 0. : 4.;
        const auto actual = fft.velocity(
            8., ly, -.2, 0., 1., 2., zero, zero, zero, add, xs, ys);
        for (std::size_t x = 0; x < xs.size(); ++x) {
            std::array<std::vector<double>, 3> coeff;
            for (auto& values : coeff) values.resize(sy);
            modes_hosgrid::nwt_additional_coefficients(
                sy, count, 8., ly, xs[x] * (8. / (nx - 1)), -.2, add,
                coeff[0].data(), coeff[1].data(), coeff[2].data());
            for (std::size_t y = 0; y < ys.size(); ++y)
                for (int c = 0; c < 3; ++c) {
                    double expected = 0.;
                    for (int j = 0; j < sy; ++j) {
                        if (c == 1 && sy > 1 && (j == 0 || j == sy - 1))
                            continue;
                        const double angle =
                            sy == 1 ? 0. : M_PI * j * ys[y] / (ny - 1);
                        expected +=
                            coeff[c][j] * (c == 1 && sy > 1 ? std::sin(angle)
                                                            : std::cos(angle));
                    }
                    EXPECT_NEAR(
                        actual[c][y * xs.size() + x], expected / 2., 2e-13);
                }
        }
        EXPECT_LE(fft.stats().scratch_bytes, 512ULL * 1024 * 1024);
    }
}

TEST(NwtReconstruction, ForcedBudgetAndVelocityValidation)
{
    std::vector<double> modes(5 * 9, .1);
    NwtReconstruction full(5, 9, 257, 513, {"full", 128 * 1024});
    EXPECT_THROW(
        full.evaluate(modes, true, true, {0}, {0}), std::runtime_error);
    NwtReconstruction tiny(5, 9, 257, 513, {"auto", 1});
    EXPECT_THROW(
        tiny.velocity(
            8., 4., 0., 0., 1., 1., modes, modes, modes, {}, {0}, {0}),
        std::runtime_error);
    NwtReconstruction fft(5, 9, 17, 33);
    EXPECT_THROW(
        fft.velocity(8., 4., 0., 0., 1., 0., modes, modes, modes, {}, {0}, {0}),
        std::invalid_argument);
    EXPECT_THROW(
        fft.velocity(
            8., 4., 0., 0., 1., 1., modes, modes, modes, {1.}, {0}, {0}),
        std::invalid_argument);
    EXPECT_THROW(
        fft.velocity(
            8., 4., 0., 0., 1., 1., modes, modes, modes, {}, {33}, {0}),
        std::invalid_argument);
}

double interpolation_error(int factor)
{
    const int sy = 5, sx = 9, ny = factor * (sy - 1) + 1,
              nx = factor * (sx - 1) + 1;
    std::vector<double> modes(sy * sx);
    modes[2 * sx + 3] = 1.;
    std::vector<int> xs(nx), ys(ny);
    std::iota(xs.begin(), xs.end(), 0);
    std::iota(ys.begin(), ys.end(), 0);
    NwtReconstruction fft(sy, sx, ny, nx, {"pruned"});
    const auto host = fft.evaluate(modes, true, true, xs, ys);
    amrex::Gpu::DeviceVector<amrex::Real> eta(host.size());
    amrex::Gpu::copy(
        amrex::Gpu::hostToDevice, host.begin(), host.end(), eta.begin());
    amrex::BoxArray boxes(
        amrex::Box(amrex::IntVect(0), amrex::IntVect(63, 31, 0)));
    amrex::DistributionMapping dm(boxes);
    amrex::MultiFab field(boxes, dm, 1, 0);
    const amrex::GpuArray<amrex::Real, 3> plo{0., 0., 0.},
        dx{M_PI / 64., M_PI / 32., 1.};
    interp_to_mfab::interp_eta_to_levelset_multifab(
        nx, ny, M_PI / (nx - 1), M_PI / (ny - 1), 0., 0., 0., false, eta, field,
        plo, dx);
    for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
        const auto values = field.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                values(i, j, k) -= std::cos(3. * (i + .5) * dx[0]) *
                                       std::cos(2. * (j + .5) * dx[1]) -
                                   .5;
            });
    }
    return field.norm0();
}

TEST(NwtReconstruction, PaddingImprovesOffGridCubicAccuracy)
{
    const auto coarse = interpolation_error(1);
    const auto fine = interpolation_error(2);
    const auto finer = interpolation_error(4);
    EXPECT_LT(fine, coarse * .2);
    EXPECT_LT(finer, fine * .15);
}
} // namespace
