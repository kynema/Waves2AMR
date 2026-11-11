#ifndef OCEAN_PADDING_H
#define OCEAN_PADDING_H

#include <array>
#include <complex>
#include <vector>

namespace modes_hosgrid {

// Dimensions use FFTW order: full (y) dimension, then half-spectrum (x)
// dimension.
std::array<int, 2> ocean_fft_dimensions(
    int ny,
    int nx,
    double ly,
    double lx,
    double dy,
    double dx,
    int max_padding_level = -1);

// Retains the HOS coefficient normalization consumed by do_ifftw().
std::vector<std::complex<double>> pad_ocean_modes(
    int ny,
    int nx,
    int padded_ny,
    int padded_nx,
    const std::vector<std::complex<double>>& modes);

} // namespace modes_hosgrid

#endif
