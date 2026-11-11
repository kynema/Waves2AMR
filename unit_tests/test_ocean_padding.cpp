#include "ocean_padding.h"
#include "modes_hosgrid.h"
#include "gtest/gtest.h"
#include <fftw3.h>
#include <cmath>
#include <stdexcept>

namespace {

TEST(OceanPadding, ResolutionAndCaps)
{
    using modes_hosgrid::ocean_fft_dimensions;
    EXPECT_EQ(
        (std::array<int, 2>{8, 8}), ocean_fft_dimensions(8, 8, 8, 8, 2, 2));
    EXPECT_EQ(
        (std::array<int, 2>{8, 8}),
        ocean_fft_dimensions(8, 8, 8, 8, .2, .4, 0));
    EXPECT_EQ(
        (std::array<int, 2>{16, 16}),
        ocean_fft_dimensions(8, 8, 8, 8, .2, .4, 1));
    EXPECT_EQ(
        (std::array<int, 2>{32, 32}),
        ocean_fft_dimensions(8, 8, 8, 8, .2, .4, 2));
    EXPECT_EQ(
        (std::array<int, 2>{64, 32}), ocean_fft_dimensions(8, 8, 8, 8, .2, .4));
    EXPECT_EQ(
        (std::array<int, 2>{1, 16}), ocean_fft_dimensions(1, 8, 1, 8, 1, .5));
    EXPECT_EQ(
        (std::array<int, 2>{16, 16}),
        ocean_fft_dimensions(8, 8, 8 * (1 + 1e-11), 8 * (1 + 1e-11), .5, .5));
    EXPECT_THROW(ocean_fft_dimensions(8, 8, 8, 8, 0, 1), std::invalid_argument);
    EXPECT_THROW(
        ocean_fft_dimensions(8, 8, 8, 8, 1, 1, -2), std::invalid_argument);
    EXPECT_THROW(
        ocean_fft_dimensions(8, 8, 8, 8, 1e-12, 1e-12), std::overflow_error);
}

std::vector<double>
reconstruct(int ny, int nx, const std::vector<std::complex<double>>& modes)
{
    auto* in = fftw_alloc_complex(ny * (nx / 2 + 1));
    std::vector<double> out(ny * nx);
    auto plan = fftw_plan_dft_c2r_2d(ny, nx, in, out.data(), FFTW_ESTIMATE);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i <= nx / 2; ++i) {
            const int index = j * (nx / 2 + 1) + i;
            const auto value = modes[index] * (i == 0 ? 1. : .5);
            in[index][0] = value.real();
            in[index][1] = value.imag();
        }
    }
    fftw_execute_dft_c2r(plan, in, out.data());
    fftw_destroy_plan(plan);
    fftw_free(in);
    return out;
}

TEST(OceanPadding, AnalyticObliqueWavesOddEvenAndAnisotropic)
{
    for (const int ny : {5, 6}) {
        for (const int nx : {7, 8}) {
            std::vector<std::complex<double>> modes(ny * (nx / 2 + 1));
            modes[0] = 2.;
            modes[(ny - 2) * (nx / 2 + 1) + 2] = {1.2, -.7};
            // Highest positive frequencies on odd grids are not Nyquist modes.
            if (ny % 2 && nx % 2) {
                modes[(ny / 2) * (nx / 2 + 1) + nx / 2] = {.3, .2};
            }
            for (const auto shape :
                 {std::array<int, 2>{ny, nx}, std::array<int, 2>{ny * 2, nx},
                  std::array<int, 2>{ny, nx * 4},
                  std::array<int, 2>{ny * 2, nx * 4}}) {
                const auto padded = modes_hosgrid::pad_ocean_modes(
                    ny, nx, shape[0], shape[1], modes);
                const auto values = reconstruct(shape[0], shape[1], padded);
                for (int j = 0; j < shape[0]; ++j) {
                    for (int i = 0; i < shape[1]; ++i) {
                        const double x = 2 * M_PI * i / shape[1];
                        const double y = 2 * M_PI * j / shape[0];
                        double exact = 2 + 1.2 * std::cos(2 * x - 2 * y) +
                                       .7 * std::sin(2 * x - 2 * y);
                        if (ny % 2 && nx % 2) {
                            const double phase = (nx / 2) * x + (ny / 2) * y;
                            exact +=
                                .3 * std::cos(phase) - .2 * std::sin(phase);
                        }
                        EXPECT_NEAR(values[j * shape[1] + i], exact, 2e-14);
                    }
                }
            }
        }
    }
}

TEST(OceanPadding, NyquistSplittingPreservesOriginalNodesAndContinuousCosines)
{
    const int ny = 6, nx = 8;
    std::vector<std::complex<double>> modes(ny * (nx / 2 + 1));
    modes[nx / 2] = 4.; // do_ifftw halves even the original compressed Nyquist.
    modes[(ny / 2) * (nx / 2 + 1)] = 3.;
    modes[(ny / 2) * (nx / 2 + 1) + nx / 2] = 2.;
    const auto original = reconstruct(ny, nx, modes);
    const auto padded =
        modes_hosgrid::pad_ocean_modes(ny, nx, ny * 2, nx * 4, modes);
    const auto values = reconstruct(ny * 2, nx * 4, padded);
    for (int j = 0; j < ny * 2; ++j) {
        for (int i = 0; i < nx * 4; ++i) {
            const double cx = std::cos(M_PI * i / 4);
            const double cy = std::cos(M_PI * j / 2);
            EXPECT_NEAR(
                values[j * nx * 4 + i], 2 * cx + 3 * cy + cx * cy, 3e-14);
            if (j % 2 == 0 && i % 4 == 0) {
                EXPECT_NEAR(
                    values[j * nx * 4 + i], original[(j / 2) * nx + i / 4],
                    1e-14);
            }
        }
    }
}

TEST(OceanPadding, InvalidShapesAndIdentity)
{
    std::vector<std::complex<double>> modes(6 * 5, {.2, -.1});
    EXPECT_EQ(modes_hosgrid::pad_ocean_modes(6, 8, 6, 8, modes), modes);
    EXPECT_THROW(
        modes_hosgrid::pad_ocean_modes(6, 8, 4, 8, modes),
        std::invalid_argument);
    modes.pop_back();
    EXPECT_THROW(
        modes_hosgrid::pad_ocean_modes(6, 8, 12, 16, modes),
        std::invalid_argument);
}

TEST(OceanPadding, PaddedElevationAndVelocityAtDepth)
{
    const int ny = 6, nx = 8, py = 12, px = 32;
    std::vector<std::complex<double>> modes(ny * (nx / 2 + 1));
    modes[(ny - 1) * (nx / 2 + 1) + 2] = {1.2, -.7};
    const auto padded = modes_hosgrid::pad_ocean_modes(ny, nx, py, px, modes);
    auto* eta = modes_hosgrid::allocate_complex(py, px);
    auto* u = modes_hosgrid::allocate_complex(py, px);
    auto* v = modes_hosgrid::allocate_complex(py, px);
    auto* w = modes_hosgrid::allocate_complex(py, px);
    std::vector<double> scratch(py * px);
    std::vector<fftw_plan> plans{modes_hosgrid::plan_ifftw(
        py, px, eta, scratch.data(), modes_hosgrid::planner_flags::estimate)};
    amrex::Gpu::DeviceVector<amrex::Real> elevation(py * px);
    amrex::Gpu::DeviceVector<amrex::Real> vel_u(py * px * 2, 0.);
    amrex::Gpu::DeviceVector<amrex::Real> vel_v(py * px * 2, 0.);
    amrex::Gpu::DeviceVector<amrex::Real> vel_w(py * px * 2, 0.);
    modes_hosgrid::copy_complex(py, px, padded, eta);
    modes_hosgrid::populate_hos_eta(py, px, 2., plans, eta, elevation);
    modes_hosgrid::populate_hos_vel(
        py, px, 16., 16., 200., -.8, 0., 2., 4., padded, padded, padded, plans,
        u, v, w, vel_u, vel_v, vel_w, py * px);
    std::vector<amrex::Real> heights(py * px), velocities(py * px * 2);
    amrex::Gpu::copy(
        amrex::Gpu::deviceToHost, elevation.begin(), elevation.end(),
        heights.begin());
    const double factor = .5 * std::exp(-std::sqrt(5.) * 2 * M_PI / 8 * .4);
    for (const auto* field : {&vel_u, &vel_v, &vel_w}) {
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, field->begin(), field->end(),
            velocities.begin());
        for (int j = 0; j < py; ++j) {
            for (int i = 0; i < px; ++i) {
                const int index = j * px + i;
                const double phase = 2 * M_PI * (2. * i / px - 1. * j / py);
                const double exact =
                    1.2 * std::cos(phase) + .7 * std::sin(phase);
                EXPECT_NEAR(heights[index], 2 * exact, 2e-14);
                EXPECT_EQ(velocities[index], 0.);
                EXPECT_NEAR(velocities[py * px + index], factor * exact, 2e-14);
            }
        }
    }
    fftw_destroy_plan(plans[0]);
    delete[] eta;
    delete[] u;
    delete[] v;
    delete[] w;
}

} // namespace
