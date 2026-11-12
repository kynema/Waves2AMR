#include "ocean_reconstruction.h"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

int main(int argc, char** argv)
{
    try {
        const int source = argc > 1 ? std::stoi(argv[1]) : 256;
        const int padded = argc > 2 ? std::stoi(argv[2]) : 4096;
        const int local = argc > 3 ? std::stoi(argv[3]) : 512;
        const int repeats = argc > 4 ? std::stoi(argv[4]) : 10;
        if (source < 2 || padded < source || local < 1 || local > padded ||
            repeats < 1)
            throw std::invalid_argument(
                "expected 2 <= source <= padded, 1 <= local <= padded, repeats "
                ">= 1");
        std::vector<std::complex<double>> modes(
            static_cast<std::size_t>(source) * (source / 2 + 1));
        for (std::size_t i = 0; i < modes.size(); ++i)
            modes[i] = {
                std::sin(.3 * i) / modes.size(),
                std::cos(.7 * i) / modes.size()};
        std::vector<int> xs(local), ys(local);
        std::iota(xs.begin(), xs.end(), 0);
        std::iota(ys.begin(), ys.end(), 0);
        std::vector<double> reference;
        std::cout << "requested,selected,scratch_bytes,seconds_per_plane,max_"
                     "abs_difference\n";
        for (const std::string method : {"full", "pruned", "auto"}) {
            modes_hosgrid::OceanReconstruction fft(
                source, source, padded, padded, {method});
            auto output = fft.evaluate(modes, xs, ys);
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < repeats; ++i)
                output = fft.evaluate(modes, xs, ys);
            const auto seconds = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - start)
                                     .count() /
                                 repeats;
            double error = 0.;
            if (reference.empty()) reference = output;
            for (std::size_t i = 0; i < output.size(); ++i)
                error = std::max(error, std::abs(output[i] - reference[i]));
            if (!std::isfinite(error) || error > 1e-10)
                throw std::runtime_error(
                    "reconstruction differs from full FFT reference");
            std::cout << method << ',' << fft.stats().method << ','
                      << fft.stats().scratch_bytes << ','
                      << std::setprecision(10) << seconds << ',' << error
                      << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
