#include "ocean_reconstruction.h"
#include "ocean_padding.h"
#include "modes_hosgrid.h"
#include "gtest/gtest.h"
#include <fftw3.h>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace {

std::vector<double> reference(
    int sy,
    int sx,
    int ny,
    int nx,
    const std::vector<std::complex<double>>& modes)
{
    const auto padded = modes_hosgrid::pad_ocean_modes(sy, sx, ny, nx, modes);
    auto* input = fftw_alloc_complex(padded.size());
    std::vector<double> output(static_cast<std::size_t>(ny) * nx);
    auto plan =
        fftw_plan_dft_c2r_2d(ny, nx, input, output.data(), FFTW_ESTIMATE);
    for (std::size_t i = 0; i < padded.size(); ++i) {
        input[i][0] = padded[i].real();
        input[i][1] = padded[i].imag();
    }
    modes_hosgrid::do_ifftw(ny, nx, plan, input, output.data());
    fftw_destroy_plan(plan);
    fftw_free(input);
    return output;
}

TEST(OceanReconstruction, OddEvenNyquistAnisotropicAndSingleRow)
{
    for (int sy : {1, 5, 6}) {
        for (int sx : {7, 8}) {
            std::vector<std::complex<double>> modes(sy * (sx / 2 + 1));
            for (std::size_t i = 0; i < modes.size(); ++i)
                modes[i] = {std::sin(.7 * i), std::cos(.3 * i)};
            for (auto shape : {std::pair<int, int>{sy, sx}, {sy * 2, sx * 4}}) {
                const int ny = shape.first, nx = shape.second;
                const auto expected = reference(sy, sx, ny, nx, modes);
                for (const std::string method : {"full", "pruned", "auto"}) {
                    modes_hosgrid::OceanReconstruction fft(
                        sy, sx, ny, nx, {method});
                    // Opposite aspect ratios exercise both transform orders.
                    for (int direction : {0, 1}) {
                        std::vector<int> xs(direction ? nx : 2),
                            ys(direction ? 1 : ny);
                        std::iota(xs.begin(), xs.end(), 0);
                        std::iota(ys.begin(), ys.end(), 0);
                        xs.back() = nx - 1;
                        const auto actual = fft.evaluate(modes, xs, ys);
                        for (std::size_t j = 0; j < ys.size(); ++j)
                            for (std::size_t i = 0; i < xs.size(); ++i)
                                EXPECT_NEAR(
                                    actual[j * xs.size() + i],
                                    expected[ys[j] * nx + xs[i]], 1e-12);
                    }
                }
            }
        }
    }
}

TEST(OceanReconstruction, BudgetAndInvalidInputs)
{
    using modes_hosgrid::OceanReconstruction;
    std::vector<std::complex<double>> modes(8 * 5, {1., 0.});
    OceanReconstruction fft(8, 8, 512, 512, {"auto", 128 * 1024});
    fft.evaluate(modes, {0, 1, 511}, {0, 1, 511});
    EXPECT_NE(fft.stats().method, "full");
    EXPECT_LE(fft.stats().scratch_bytes, 128 * 1024);
    EXPECT_THROW(fft.evaluate(modes, {0}, {0}, 128 * 1024), std::runtime_error);
    OceanReconstruction full(8, 8, 512, 512, {"full", 128 * 1024});
    EXPECT_THROW(full.evaluate(modes, {0}, {0}), std::runtime_error);
    OceanReconstruction tiny(8, 8, 512, 512, {"auto", 1});
    EXPECT_THROW(tiny.evaluate(modes, {0}, {0}), std::runtime_error);
    EXPECT_THROW(fft.evaluate(modes, {1, 0}, {0}), std::invalid_argument);
    EXPECT_THROW(fft.evaluate(modes, {512}, {0}), std::invalid_argument);
    EXPECT_THROW(fft.evaluate({}, {0}, {0}), std::invalid_argument);
    EXPECT_TRUE(fft.evaluate(modes, {}, {}).empty());
    EXPECT_THROW((OceanReconstruction(8, 8, 4, 4)), std::invalid_argument);
}

TEST(OceanReconstruction, CachedSelectionAndPlannerPreserveInput)
{
    const std::vector<std::complex<double>> modes(6 * 5, {.3, -.7});
    for (const std::string method : {"full", "pruned", "auto"}) {
        modes_hosgrid::OceanReconstruction fft(
            6, 8, 24, 32, {method, 512 * 1024, FFTW_MEASURE});
        const auto first = fft.evaluate(modes, {0, 3, 31}, {0, 23});
        EXPECT_EQ(fft.stats().selections, 1);
        EXPECT_EQ(first, fft.evaluate(modes, {0, 3, 31}, {0, 23}));
        EXPECT_EQ(fft.stats().selections, 1);
        fft.evaluate(modes, {0, 4, 31}, {0, 23});
        EXPECT_EQ(fft.stats().selections, 2);
        fft.evaluate(modes, {0, 4, 31}, {0, 23}, 1024);
        EXPECT_EQ(fft.stats().selections, 3);
        for (auto value : modes)
            EXPECT_EQ(value, (std::complex<double>{.3, -.7}));
    }
}

TEST(OceanReconstruction, BothSeparableOrdersMatchFullReference)
{
    const int source = 8, padded = 256;
    std::vector<std::complex<double>> modes(source * (source / 2 + 1));
    for (std::size_t n = 0; n < modes.size(); ++n)
        modes[n] = {std::sin(.1 * n), std::cos(.7 * n)};
    const auto expected = reference(source, source, padded, padded, modes);
    for (bool x_first : {false, true}) {
        std::vector<int> xs(x_first ? 1 : padded), ys(x_first ? padded : 1);
        std::iota(xs.begin(), xs.end(), 0);
        std::iota(ys.begin(), ys.end(), 0);
        modes_hosgrid::OceanReconstruction fft(
            source, source, padded, padded, {"pruned"});
        const auto actual = fft.evaluate(modes, xs, ys);
        EXPECT_EQ(fft.stats().method, x_first ? "pruned-x" : "pruned-y");
        for (std::size_t j = 0; j < ys.size(); ++j)
            for (std::size_t i = 0; i < xs.size(); ++i)
                EXPECT_NEAR(
                    actual[j * xs.size() + i], expected[ys[j] * padded + xs[i]],
                    1e-12);
    }
}

TEST(OceanReconstruction, CompactDepthCoefficientsMatchPaddedVelocity)
{
    const int sy = 6, sx = 8, ny = 24, nx = 32;
    std::vector<std::complex<double>> source(sy * (sx / 2 + 1));
    for (std::size_t i = 0; i < source.size(); ++i)
        source[i] = {std::sin(.4 * i) * .01, std::cos(.2 * i) * .01};
    auto padded = modes_hosgrid::pad_ocean_modes(sy, sx, ny, nx, source);
    auto* a = modes_hosgrid::allocate_complex(ny, nx);
    auto* b = modes_hosgrid::allocate_complex(ny, nx);
    auto* c = modes_hosgrid::allocate_complex(ny, nx);
    auto* out = modes_hosgrid::allocate_real(ny, nx);
    auto plan = modes_hosgrid::plan_ifftw(
        ny, nx, a, out, modes_hosgrid::planner_flags::estimate);
    amrex::Gpu::DeviceVector<amrex::Real> u(nx * ny), v(nx * ny), w(nx * ny);
    std::vector<int> xs(nx), ys(ny);
    std::iota(xs.begin(), xs.end(), 0);
    std::iota(ys.begin(), ys.end(), 0);
    for (double z : {-10., -4., 0., 2.}) {
        modes_hosgrid::populate_hos_vel(
            ny, nx, 20., 15., 10., z, 0., 2., 3., padded, padded, padded,
            {plan}, a, b, c, u, v, w);
        std::vector<std::complex<double>> um, vm, wm;
        modes_hosgrid::ocean_velocity_modes(
            sy, sx, 20., 15., 10., z, 0., 2., 3., source, source, source, um,
            vm, wm);
        modes_hosgrid::OceanReconstruction fft(sy, sx, ny, nx, {"pruned"});
        const auto actual = fft.evaluate(wm, xs, ys);
        std::vector<amrex::Real> expected(w.size());
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, w.begin(), w.end(), expected.begin());
        for (std::size_t i = 0; i < actual.size(); ++i)
            EXPECT_NEAR(actual[i], expected[i], 1e-12);
        const auto au = fft.evaluate(um, xs, ys);
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, u.begin(), u.end(), expected.begin());
        for (std::size_t i = 0; i < au.size(); ++i)
            EXPECT_NEAR(au[i], expected[i], 1e-12);
    }
    fftw_destroy_plan(plan);
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] out;
}
} // namespace
