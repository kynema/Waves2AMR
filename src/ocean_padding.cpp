#include "ocean_padding.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

int padded_dimension(int n, double length, double spacing, int cap)
{
    if (n < 1 || !std::isfinite(length) || length <= 0 ||
        !std::isfinite(spacing) || spacing <= 0 || cap < -1) {
        throw std::invalid_argument(
            "Waves2AMR: invalid FFT padding dimensions, spacing or cap");
    }
    int level = 0;
    // HOS ASCII metadata is rounded; do not double a grid for that roundoff
    // alone.
    while ((cap == -1 || level < cap) && length / n > spacing * (1 + 1e-9)) {
        if (n > std::numeric_limits<int>::max() / 2) {
            throw std::overflow_error(
                "Waves2AMR: padded FFT dimension exceeds integer range");
        }
        n *= 2;
        ++level;
    }
    return n;
}

void check_size(int ny, int nx)
{
    if (ny < 1 || nx < 1 || ny > std::numeric_limits<int>::max() / nx) {
        throw std::overflow_error(
            "Waves2AMR: FFT grid exceeds supported integer range");
    }
}

} // namespace

std::array<int, 2> modes_hosgrid::ocean_fft_dimensions(
    int ny,
    int nx,
    double ly,
    double lx,
    double dy,
    double dx,
    int max_padding_level)
{
    const int py = padded_dimension(ny, ly, dy, max_padding_level);
    const int px = padded_dimension(nx, lx, dx, max_padding_level);
    check_size(py, px);
    return {py, px};
}

std::vector<std::complex<double>> modes_hosgrid::pad_ocean_modes(
    int ny,
    int nx,
    int padded_ny,
    int padded_nx,
    const std::vector<std::complex<double>>& modes)
{
    check_size(ny, nx);
    check_size(padded_ny, padded_nx);
    if (padded_ny < ny || padded_nx < nx ||
        modes.size() != static_cast<std::size_t>(ny) * (nx / 2 + 1)) {
        throw std::invalid_argument(
            "Waves2AMR: incompatible zero-padding dimensions or mode count");
    }
    std::vector<std::complex<double>> padded(
        static_cast<std::size_t>(padded_ny) * (padded_nx / 2 + 1), 0.);
    for (int j = 0; j < ny; ++j) {
        const int ky = j <= ny / 2 ? j : j - ny;
        const int dest_y = ky < 0 ? padded_ny + ky : ky;
        const bool split_y = ny % 2 == 0 && j == ny / 2 && padded_ny > ny;
        for (int i = 0; i <= nx / 2; ++i) {
            const bool split_x = nx % 2 == 0 && i == nx / 2 && padded_nx > nx;
            // Split old Nyquist modes between the now-distinct +/- frequencies.
            const auto value = modes[j * (nx / 2 + 1) + i] *
                               (split_y ? 0.5 : 1.) * (split_x ? 0.5 : 1.);
            padded[dest_y * (padded_nx / 2 + 1) + i] += value;
            if (split_y) {
                padded[(padded_ny - ky) * (padded_nx / 2 + 1) + i] += value;
            }
        }
    }
    return padded;
}
