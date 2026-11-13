#include "ocean_reconstruction.h"
#include "modes_hosgrid.h"
#include <fftw3.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

using Complex = std::complex<double>;

std::size_t checked_product(std::size_t a, std::size_t b)
{
    if (b && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error("Waves2AMR: reconstruction size overflow");
    }
    return a * b;
}

std::vector<int> frequency_bins(int source, int padded)
{
    std::vector<int> bins;
    for (int k = -source / 2; k <= source / 2; ++k) {
        bins.push_back(k < 0 ? padded + k : k);
    }
    std::sort(bins.begin(), bins.end());
    bins.erase(std::unique(bins.begin(), bins.end()), bins.end());
    return bins;
}

int position(const std::vector<int>& bins, int bin)
{
    return static_cast<int>(
        std::lower_bound(bins.begin(), bins.end(), bin) - bins.begin());
}

struct Workspace
{
    fftw_complex* row{nullptr};
    fftw_complex* half{nullptr};
    double* plane{nullptr};
    fftw_plan px{nullptr}, py{nullptr}, full{nullptr};
    std::vector<Complex> intermediate;
    ~Workspace()
    {
        if (px) fftw_destroy_plan(px);
        if (py) fftw_destroy_plan(py);
        if (full) fftw_destroy_plan(full);
        fftw_free(row);
        fftw_free(half);
        fftw_free(plane);
    }
};

void validate_indices(const std::vector<int>& indices, int size)
{
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= size ||
            (i && indices[i] <= indices[i - 1])) {
            throw std::invalid_argument(
                "Waves2AMR: reconstruction indices must be sorted, unique and "
                "in range");
        }
    }
}

} // namespace

struct modes_hosgrid::OceanReconstruction::Impl
{
    int sy, sx, ny, nx;
    OceanReconstructionOptions options;
    OceanReconstructionStats stats;
    std::vector<int> kx, ky, xs, ys;
    std::unique_ptr<Workspace> work;
    std::vector<Complex> spectrum;
    std::size_t reserved_bytes{0};

    Impl(
        int source_y,
        int source_x,
        int y,
        int x,
        OceanReconstructionOptions opts)
        : sy(source_y), sx(source_x), ny(y), nx(x), options(std::move(opts))
    {
        if (sy < 1 || sx < 2 || ny < sy || nx < sx ||
            options.scratch_bytes == 0 ||
            (options.method != "auto" && options.method != "full" &&
             options.method != "pruned")) {
            throw std::invalid_argument(
                "Waves2AMR: invalid ocean reconstruction options");
        }
        if (checked_product(static_cast<std::size_t>(nx), ny) >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                "Waves2AMR: reconstruction grid exceeds supported integer "
                "range");
        }
        kx = frequency_bins(sx, nx);
        ky = frequency_bins(sy, ny);
    }

    std::size_t bytes(const std::string& method) const
    {
        // Include compact spectrum, returned host samples and their device
        // copy.
        auto count = checked_product(kx.size(), ky.size()) * sizeof(Complex);
        count += checked_product(xs.size(), ys.size()) * 2 * sizeof(double);
        if (method == "full") {
            count += checked_product(ny, nx / 2 + 1) * sizeof(Complex);
            count += checked_product(ny, nx) * sizeof(double);
        } else {
            const auto middle = method == "pruned-x"
                                    ? checked_product(ky.size(), xs.size())
                                    : checked_product(kx.size(), ys.size());
            count += (middle + static_cast<std::size_t>(std::max(nx, ny))) *
                     sizeof(Complex);
        }
        if (count > std::numeric_limits<std::size_t>::max() - reserved_bytes)
            throw std::overflow_error(
                "Waves2AMR: reconstruction budget overflow");
        return count + reserved_bytes;
    }

    double cost(const std::string& method) const
    {
        const auto cx = nx * std::log2(std::max(nx, 2));
        const auto cy = ny * std::log2(std::max(ny, 2));
        if (method == "full") return ny * cx + nx * cy;
        if (method == "pruned-x") return ky.size() * cx + xs.size() * cy;
        return kx.size() * cy + ys.size() * cx;
    }

    void allocate(const std::string& method)
    {
        work = std::make_unique<Workspace>();
        if (method == "full") {
            work->half = fftw_alloc_complex(checked_product(ny, nx / 2 + 1));
            work->plane = fftw_alloc_real(checked_product(ny, nx));
            if (!work->half || !work->plane) throw std::bad_alloc();
            work->full = fftw_plan_dft_c2r_2d(
                ny, nx, work->half, work->plane, options.planner_flags);
            if (!work->full)
                throw std::runtime_error("Waves2AMR: full FFT plan failed");
        } else {
            work->row = fftw_alloc_complex(std::max(nx, ny));
            if (!work->row) throw std::bad_alloc();
            work->px = fftw_plan_dft_1d(
                nx, work->row, work->row, FFTW_BACKWARD, options.planner_flags);
            work->py = fftw_plan_dft_1d(
                ny, work->row, work->row, FFTW_BACKWARD, options.planner_flags);
            if (!work->px || !work->py)
                throw std::runtime_error("Waves2AMR: pruned FFT plan failed");
            work->intermediate.resize(
                method == "pruned-x" ? checked_product(ky.size(), xs.size())
                                     : checked_product(kx.size(), ys.size()));
        }
    }

    void prepare(const std::vector<Complex>& modes)
    {
        spectrum.assign(checked_product(kx.size(), ky.size()), {});
        const auto add = [&](int y, int x, Complex value) {
            spectrum[position(ky, y) * kx.size() + position(kx, x)] += value;
        };
        for (int j = 0; j < sy; ++j) {
            const int signed_y = j <= sy / 2 ? j : j - sy;
            const int y = signed_y < 0 ? ny + signed_y : signed_y;
            const bool split_y = sy % 2 == 0 && j == sy / 2 && ny > sy;
            for (int i = 0; i <= sx / 2; ++i) {
                const bool split_x = sx % 2 == 0 && i == sx / 2 && nx > sx;
                auto value = modes[j * (sx / 2 + 1) + i] * (split_y ? .5 : 1.) *
                             (split_x ? .5 : 1.) * (i == 0 ? 1. : .5);
                for (int branch = 0; branch < (split_y ? 2 : 1); ++branch) {
                    const int dest_y = branch == 0 ? y : ny - y;
                    add(dest_y, i, value);
                    if (i != 0 && !(nx % 2 == 0 && i == nx / 2)) {
                        add((ny - dest_y) % ny, nx - i, std::conj(value));
                    }
                }
            }
        }
    }

    std::vector<double> execute(const std::string& method)
    {
        std::vector<double> output(checked_product(xs.size(), ys.size()));
        if (method == "full") {
            const auto count = checked_product(ny, nx / 2 + 1);
            for (std::size_t i = 0; i < count; ++i) {
                work->half[i][0] = work->half[i][1] = 0.;
            }
            for (std::size_t j = 0; j < ky.size(); ++j) {
                for (std::size_t i = 0; i < kx.size(); ++i) {
                    if (kx[i] > nx / 2) continue;
                    const auto value = spectrum[j * kx.size() + i];
                    auto& dest =
                        work->half
                            [static_cast<std::size_t>(ky[j]) * (nx / 2 + 1) +
                             kx[i]];
                    dest[0] = value.real();
                    dest[1] = value.imag();
                }
            }
            fftw_execute(work->full);
            for (std::size_t j = 0; j < ys.size(); ++j)
                for (std::size_t i = 0; i < xs.size(); ++i)
                    output[j * xs.size() + i] =
                        work->plane
                            [static_cast<std::size_t>(ys[j]) * nx + xs[i]];
            return output;
        }
        const bool x_first = method == "pruned-x";
        const auto& first_bins = x_first ? kx : ky;
        const auto& second_bins = x_first ? ky : kx;
        const auto& first_points = x_first ? xs : ys;
        const auto& second_points = x_first ? ys : xs;
        const int first_size = x_first ? nx : ny;
        const int second_size = x_first ? ny : nx;
        const auto first_plan = x_first ? work->px : work->py;
        const auto second_plan = x_first ? work->py : work->px;
        for (std::size_t b = 0; b < second_bins.size(); ++b) {
            for (int n = 0; n < first_size; ++n)
                work->row[n][0] = work->row[n][1] = 0.;
            for (std::size_t a = 0; a < first_bins.size(); ++a) {
                const auto value = x_first ? spectrum[b * kx.size() + a]
                                           : spectrum[a * kx.size() + b];
                work->row[first_bins[a]][0] = value.real();
                work->row[first_bins[a]][1] = value.imag();
            }
            fftw_execute(first_plan);
            for (std::size_t p = 0; p < first_points.size(); ++p) {
                const auto& value = work->row[first_points[p]];
                work->intermediate[b * first_points.size() + p] = {
                    value[0], value[1]};
            }
        }
        for (std::size_t p = 0; p < first_points.size(); ++p) {
            for (int n = 0; n < second_size; ++n)
                work->row[n][0] = work->row[n][1] = 0.;
            for (std::size_t b = 0; b < second_bins.size(); ++b) {
                const auto value =
                    work->intermediate[b * first_points.size() + p];
                work->row[second_bins[b]][0] = value.real();
                work->row[second_bins[b]][1] = value.imag();
            }
            fftw_execute(second_plan);
            for (std::size_t q = 0; q < second_points.size(); ++q) {
                const auto index =
                    x_first ? q * xs.size() + p : p * xs.size() + q;
                output[index] = work->row[second_points[q]][0];
            }
        }
        return output;
    }

    std::vector<std::string> feasible_methods() const
    {
        std::vector<std::string> candidates;
        for (const std::string method : {"full", "pruned-x", "pruned-y"}) {
            if ((options.method == "full" && method != "full") ||
                (options.method == "pruned" && method == "full"))
                continue;
            if (bytes(method) <= options.scratch_bytes)
                candidates.push_back(method);
        }
        if (candidates.empty()) {
            throw std::runtime_error(
                "Waves2AMR: no reconstruction method fits "
                "fft_scratch_memory_mb; "
                "increase the budget or reduce padding/local horizontal "
                "footprint");
        }
        return candidates;
    }

    void select(const std::vector<std::string>& candidates)
    {
        const auto cheapest = *std::min_element(
            candidates.begin(), candidates.end(),
            [&](const auto& a, const auto& b) { return cost(a) < cost(b); });
        std::string best;
        double best_time = std::numeric_limits<double>::infinity();
        for (const auto& method : candidates) {
            // Avoid expensive calibration of clearly uncompetitive candidates.
            if (cost(method) > 4 * cost(cheapest)) continue;
            allocate(method);
            execute(method);
            const auto start = std::chrono::steady_clock::now();
            for (int repeat = 0; repeat < 2; ++repeat) execute(method);
            const auto seconds = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - start)
                                     .count() /
                                 2;
            if (seconds < best_time * .9) {
                best = method;
                best_time = seconds;
            }
            work.reset();
        }
        allocate(best);
        stats = {best, bytes(best), best_time, stats.selections + 1};
    }
};

modes_hosgrid::OceanReconstruction::OceanReconstruction(
    int sy, int sx, int ny, int nx, OceanReconstructionOptions options)
    : impl(std::make_unique<Impl>(sy, sx, ny, nx, std::move(options)))
{}
modes_hosgrid::OceanReconstruction::~OceanReconstruction() = default;

std::vector<double> modes_hosgrid::OceanReconstruction::evaluate(
    const std::vector<Complex>& modes,
    const std::vector<int>& xs,
    const std::vector<int>& ys,
    std::size_t reserved_bytes)
{
    if (modes.size() != checked_product(impl->sy, impl->sx / 2 + 1))
        throw std::invalid_argument(
            "Waves2AMR: incompatible compact mode count");
    validate_indices(xs, impl->nx);
    validate_indices(ys, impl->ny);
    if (xs.empty() || ys.empty()) return {};
    if (impl->xs != xs || impl->ys != ys ||
        impl->reserved_bytes != reserved_bytes) {
        impl->work.reset();
        impl->xs = xs;
        impl->ys = ys;
        impl->reserved_bytes = reserved_bytes;
    }
    const auto candidates =
        impl->work ? std::vector<std::string>{} : impl->feasible_methods();
    impl->prepare(modes);
    if (!impl->work) impl->select(candidates);
    return impl->execute(impl->stats.method);
}

const modes_hosgrid::OceanReconstructionStats&
modes_hosgrid::OceanReconstruction::stats() const
{
    return impl->stats;
}

void modes_hosgrid::ocean_velocity_modes(
    int sy,
    int sx,
    double xlen,
    double ylen,
    double depth,
    double z,
    double zsl,
    double dimL,
    double dimT,
    const std::vector<Complex>& mx,
    const std::vector<Complex>& my,
    const std::vector<Complex>& mz,
    std::vector<Complex>& u,
    std::vector<Complex>& v,
    std::vector<Complex>& w)
{
    if (sy < 1 || sx < 2)
        throw std::invalid_argument(
            "Waves2AMR: invalid compact velocity dimensions");
    const auto count = checked_product(sy, sx / 2 + 1);
    if (mx.size() != count || my.size() != count || mz.size() != count ||
        xlen <= 0 || ylen <= 0 || depth <= 0 || dimL <= 0 || dimT <= 0) {
        throw std::invalid_argument(
            "Waves2AMR: invalid compact velocity inputs");
    }
    u.resize(count);
    v.resize(count);
    w.resize(count);
    const double nd_z = (z - zsl) / dimL, nd_depth = depth / dimL;
    const double twoPi_xlen = 2.0 * M_PI / (xlen / dimL);
    const double twoPi_ylen = 2.0 * M_PI / (ylen / dimL);
    for (int j = 0; j < sy; ++j) {
        // HOS-Ocean stores x along the compact (column) index and y along rows.
        const double ky = (j <= sy / 2 ? j : j - sy) * twoPi_ylen;
        for (int i = 0; i <= sx / 2; ++i) {
            const double kx = i * twoPi_xlen;
            const double k = std::sqrt(ky * ky + kx * kx);
            const auto [a, b] = ocean_depth_factors(
                k, nd_depth, nd_z, i == 0, i == 0 && j == 0);
            const auto index = j * (sx / 2 + 1) + i;
            u[index] = a * mx[index] * (dimL / dimT);
            v[index] = a * my[index] * (dimL / dimT);
            w[index] = b * mz[index] * (dimL / dimT);
        }
    }
}

void modes_hosgrid::ocean_velocity_rate_modes(
    int sy,
    int sx,
    double xlen,
    double ylen,
    double depth,
    double dimL,
    const std::vector<Complex>& mt,
    std::vector<Complex>& mx_t,
    std::vector<Complex>& my_t,
    std::vector<Complex>& mz_t)
{
    if (sy < 1 || sx < 2)
        throw std::invalid_argument(
            "Waves2AMR: invalid compact velocity rate dimensions");
    const auto count = checked_product(sy, sx / 2 + 1);
    if (mt.size() != count || !(xlen > 0) || !(ylen > 0) || !(depth > 0) ||
        !(dimL > 0))
        throw std::invalid_argument(
            "Waves2AMR: invalid compact velocity rate inputs");
    mx_t.resize(count);
    my_t.resize(count);
    mz_t.resize(count);
    const double nd_depth = depth / dimL;
    const double twoPi_xlen = 2.0 * M_PI / (xlen / dimL);
    const double twoPi_ylen = 2.0 * M_PI / (ylen / dimL);
    const Complex imag(0., 1.);
    // HOS-Ocean stores x along the compact (column) index and y along rows.
    for (int j = 0; j < sy; ++j) {
        const double ky = (j <= sy / 2 ? j : j - sy) * twoPi_ylen;
        for (int i = 0; i <= sx / 2; ++i) {
            const double kx = i * twoPi_xlen;
            const double k = std::sqrt(kx * kx + ky * ky);
            const auto index = j * (sx / 2 + 1) + i;
            mx_t[index] = imag * kx * mt[index];
            my_t[index] = imag * ky * mt[index];
            mz_t[index] = k * std::tanh(k * nd_depth) * mt[index];
        }
    }
}
