#include "nwt_reconstruction.h"
#include "ocean_padding.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
int extension(int n)
{
    if (n < 1 || n > std::numeric_limits<int>::max() / 2)
        throw std::invalid_argument("Waves2AMR: invalid NWT dimension");
    return n == 1 ? 1 : 2 * (n - 1);
}

void check_indices(const std::vector<int>& indices, int size)
{
    for (std::size_t i = 0; i < indices.size(); ++i)
        if (indices[i] < 0 || indices[i] >= size ||
            (i && indices[i] <= indices[i - 1]))
            throw std::invalid_argument(
                "Waves2AMR: invalid NWT sample indices");
}
} // namespace

std::array<int, 2> modes_hosgrid::nwt_fft_dimensions(
    int ny, int nx, double ly, double lx, double dy, double dx, int cap)
{
    if (nx < 3 || (ny != 1 && ny < 3))
        throw std::invalid_argument(
            "Waves2AMR: NWT requires at least three endpoint samples");
    auto dims = ocean_fft_dimensions(
        ny == 1 ? 1 : ny - 1, nx - 1, ny == 1 ? 1. : ly, lx, ny == 1 ? 1. : dy,
        dx, cap);
    return {ny == 1 ? 1 : dims[0] + 1, dims[1] + 1};
}

struct modes_hosgrid::NwtReconstruction::Impl
{
    int sy, sx, ny, nx;
    OceanReconstructionOptions options;
    OceanReconstruction fft;
    double* input{nullptr};
    double* output{nullptr};
    fftw_plan cosine{nullptr}, sine{nullptr};

    Impl(
        int source_y,
        int source_x,
        int y,
        int x,
        OceanReconstructionOptions opts)
        : sy(source_y)
        , sx(source_x)
        , ny(y)
        , nx(x)
        , options(opts)
        , fft(extension(sy), extension(sx), extension(ny), extension(nx), opts)
    {
        if (sx < 3 || (sy != 1 && sy < 3) || nx < sx || ny < sy ||
            (sy == 1 && ny != 1))
            throw std::invalid_argument(
                "Waves2AMR: invalid NWT reconstruction dimensions");
    }

    ~Impl()
    {
        if (cosine) fftw_destroy_plan(cosine);
        if (sine) fftw_destroy_plan(sine);
        fftw_free(input);
        fftw_free(output);
    }

    std::size_t conversion_bytes() const
    {
        return static_cast<std::size_t>(extension(sy)) * sx *
               sizeof(std::complex<double>);
    }

    void plan_additional()
    {
        if (ny == 1 || (cosine && sine)) return;
        input = fftw_alloc_real(ny);
        output = fftw_alloc_real(ny);
        if (!input || !output) {
            fftw_free(input);
            fftw_free(output);
            input = output = nullptr;
            throw std::bad_alloc();
        }
        cosine = fftw_plan_r2r_1d(
            ny, input, output, FFTW_REDFT00, options.planner_flags);
        sine = fftw_plan_r2r_1d(
            ny - 2, input, output, FFTW_RODFT00, options.planner_flags);
        if (!cosine || !sine) {
            if (cosine) fftw_destroy_plan(cosine);
            if (sine) fftw_destroy_plan(sine);
            cosine = sine = nullptr;
            fftw_free(input);
            fftw_free(output);
            input = output = nullptr;
            throw std::runtime_error(
                "Waves2AMR: NWT wavemaker FFT plan failed");
        }
    }
};

modes_hosgrid::NwtReconstruction::NwtReconstruction(
    int sy, int sx, int ny, int nx, OceanReconstructionOptions options)
    : impl(std::make_unique<Impl>(sy, sx, ny, nx, options))
{}
modes_hosgrid::NwtReconstruction::~NwtReconstruction() = default;

std::vector<double> modes_hosgrid::NwtReconstruction::evaluate(
    const std::vector<double>& modes,
    bool cos_y,
    bool cos_x,
    const std::vector<int>& xs,
    const std::vector<int>& ys,
    std::size_t reserved_bytes)
{
    const auto& d = *impl;
    if (modes.size() != static_cast<std::size_t>(d.sy) * d.sx)
        throw std::invalid_argument("Waves2AMR: invalid NWT mode count");
    check_indices(xs, d.nx);
    check_indices(ys, d.ny);
    if (xs.empty() || ys.empty()) return {};
    const auto overhead = d.conversion_bytes() +
                          2 * static_cast<std::size_t>(d.ny) * sizeof(double);
    if (reserved_bytes > d.options.scratch_bytes ||
        overhead > d.options.scratch_bytes - reserved_bytes)
        throw std::runtime_error(
            "Waves2AMR: NWT conversion exceeds FFT scratch budget");
    // Even/odd reflection converts NWT cosine/sine modes to the shared compact
    // FFT backend. Endpoint cosine modes retain their full series amplitude;
    // endpoint sine modes are absent in the original DST-I basis.
    const int ey = extension(d.sy);
    std::vector<std::complex<double>> compact(
        static_cast<std::size_t>(ey) * d.sx);
    for (int j = 0; j < d.sy; ++j) {
        if (!cos_y && (j == 0 || j == d.sy - 1)) continue;
        for (int i = 0; i < d.sx; ++i) {
            if (!cos_x && (i == 0 || i == d.sx - 1)) continue;
            const std::complex<double> xfactor =
                cos_x ? std::complex<double>(i == d.sx - 1 ? 2. : 1.)
                      : std::complex<double>(0., -1.);
            auto value = modes[j * d.sx + i] * xfactor;
            if (j == 0 || j == d.sy - 1) {
                compact[j * d.sx + i] = value;
            } else {
                const auto yfactor = cos_y ? std::complex<double>(.5)
                                           : std::complex<double>(0., -.5);
                compact[j * d.sx + i] = value * yfactor;
                compact[(ey - j) * d.sx + i] = value * std::conj(yfactor);
            }
        }
    }
    return impl->fft.evaluate(compact, xs, ys, reserved_bytes + overhead);
}

std::array<std::vector<double>, 3> modes_hosgrid::NwtReconstruction::velocity(
    double xlen,
    double ylen,
    double z,
    double zsl,
    double dimL,
    double dimT,
    const std::vector<double>& mx,
    const std::vector<double>& my,
    const std::vector<double>& mz,
    const std::vector<double>& additional,
    const std::vector<int>& xs,
    const std::vector<int>& ys,
    std::size_t reserved_bytes)
{
    auto& d = *impl;
    const auto count = static_cast<std::size_t>(d.sy) * d.sx;
    if (mx.size() != count || my.size() != count || mz.size() != count ||
        additional.size() % d.sy || !std::isfinite(z) || !std::isfinite(zsl) ||
        !std::isfinite(xlen) || xlen <= 0 || !std::isfinite(ylen) ||
        (d.sy != 1 && ylen <= 0) || !std::isfinite(dimL) || dimL <= 0 ||
        !std::isfinite(dimT) || dimT <= 0)
        throw std::invalid_argument("Waves2AMR: invalid NWT velocity inputs");
    check_indices(xs, d.nx);
    check_indices(ys, d.ny);
    if (xs.empty() || ys.empty()) return {};
    const auto samples = xs.size() * ys.size();
    // Include all three depth arrays, the other two returned components and
    // wavemaker coefficients. Evaluate accounts the wavemaker FFT scratch
    // and the shared FFT's own result.
    const auto extra = (3 * count + 2 * samples + 3 * d.sy) * sizeof(double);
    if (reserved_bytes > d.options.scratch_bytes ||
        extra > d.options.scratch_bytes - reserved_bytes)
        throw std::runtime_error(
            "Waves2AMR: NWT velocity exceeds FFT scratch budget");
    const auto reserved = reserved_bytes + extra;
    std::array<std::vector<double>, 3> depth;
    for (auto& component : depth) component.resize(count);
    const double nd_z = (z - zsl) / dimL;
    for (int j = 0; j < d.sy; ++j)
        for (int i = 0; i < d.sx; ++i) {
            const double kx = (M_PI / (xlen / dimL)) * i;
            const double ky = d.sy == 1 ? 0. : (M_PI / (ylen / dimL)) * j;
            const auto [a, b] = nwt_depth_factors(
                std::sqrt(kx * kx + ky * ky), nd_z, i == 0, i == 0 && j == 0);
            const auto n = j * d.sx + i;
            depth[0][n] = a * mx[n];
            depth[1][n] = a * my[n];
            depth[2][n] = b * mz[n];
        }
    std::array<std::vector<double>, 3> result;
    result[0] = evaluate(depth[0], true, false, xs, ys, reserved);
    result[1] = evaluate(depth[1], false, true, xs, ys, reserved);
    result[2] = evaluate(depth[2], true, true, xs, ys, reserved);
    if (!additional.empty()) {
        // The nonperiodic wavemaker dependence in x is not a Fourier basis.
        // Evaluate it only at requested x nodes, then FFT its y coefficients.
        d.plan_additional();
        std::array<std::vector<double>, 3> coeff;
        for (auto& component : coeff) component.resize(d.sy);
        for (std::size_t ix = 0; ix < xs.size(); ++ix) {
            nwt_additional_coefficients(
                d.sy, static_cast<int>(additional.size() / d.sy), xlen / dimL,
                ylen / dimL, xs[ix] * (xlen / dimL / (d.nx - 1)), nd_z,
                additional, coeff[0].data(), coeff[1].data(), coeff[2].data());
            for (int c = 0; c < 3; ++c) {
                if (d.sy == 1) {
                    result[c][ix] += coeff[c][0];
                    continue;
                }
                std::fill(d.input, d.input + d.ny, 0.);
                if (c == 1) {
                    for (int j = 1; j < d.sy - 1; ++j)
                        d.input[j - 1] = .5 * coeff[c][j];
                    fftw_execute(d.sine);
                } else {
                    for (int j = 0; j < d.sy; ++j)
                        d.input[j] =
                            coeff[c][j] * (j == 0 || j == d.ny - 1 ? 1. : .5);
                    fftw_execute(d.cosine);
                }
                for (std::size_t iy = 0; iy < ys.size(); ++iy) {
                    const int j = ys[iy];
                    const double value =
                        c == 1
                            ? (j == 0 || j == d.ny - 1 ? 0. : d.output[j - 1])
                            : d.output[j];
                    result[c][iy * xs.size() + ix] += value;
                }
            }
        }
    }
    for (auto& component : result)
        for (auto& value : component) value *= dimL / dimT;
    return result;
}

const modes_hosgrid::OceanReconstructionStats&
modes_hosgrid::NwtReconstruction::stats() const
{
    return impl->fft.stats();
}

void modes_hosgrid::nwt_velocity_rate_modes(
    int sy,
    int sx,
    double xlen,
    double ylen,
    double dimL,
    const std::vector<double>& mt,
    std::vector<double>& mx_t,
    std::vector<double>& my_t,
    std::vector<double>& mz_t)
{
    const auto count = static_cast<std::size_t>(sy) * sx;
    if (sy < 1 || sx < 2 || mt.size() != count || !(xlen > 0) ||
        (sy != 1 && !(ylen > 0)) || !(dimL > 0))
        throw std::invalid_argument(
            "Waves2AMR: invalid NWT velocity rate inputs");
    mx_t.resize(count);
    my_t.resize(count);
    mz_t.resize(count);
    for (int j = 0; j < sy; ++j)
        for (int i = 0; i < sx; ++i) {
            const double kx = (M_PI / (xlen / dimL)) * i;
            const double ky = sy == 1 ? 0. : (M_PI / (ylen / dimL)) * j;
            const double k = std::sqrt(kx * kx + ky * ky);
            const auto n = static_cast<std::size_t>(j) * sx + i;
            mx_t[n] = -kx * mt[n];
            my_t[n] = -ky * mt[n];
            mz_t[n] = k * std::tanh(k) * mt[n];
        }
}

std::pair<double, double> modes_hosgrid::nwt_depth_factors(
    double k, double z, bool first_column, bool zero_mode)
{
    if (zero_mode) return {1., 1.};
    const double kZ = k * (z + 1.);
    double a, b;
    if (first_column) {
        if (kZ < 50. && k < 50.) {
            a = exp(k * z) * (1. + exp(-2. * kZ)) / (1. + exp(-2. * k));
            b = exp(k * z) * (1. - exp(-2. * kZ)) / (1. - exp(-2. * k));
        } else {
            a = exp(kZ);
            b = a;
        }
    } else if (kZ < 50. && k < 50.) {
        a = cosh(kZ) / cosh(k);
        b = sinh(kZ) / sinh(k);
    } else {
        a = exp(k * z);
        b = a;
    }
    return {std::min(a, 3.), std::min(b, 3.)};
}
