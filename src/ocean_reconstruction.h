#ifndef OCEAN_RECONSTRUCTION_H
#define OCEAN_RECONSTRUCTION_H

#include <complex>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <fftw3.h>

namespace modes_hosgrid {

struct OceanReconstructionOptions
{
    std::string method{"auto"};
    std::size_t scratch_bytes{512ULL * 1024 * 1024};
    unsigned planner_flags{FFTW_ESTIMATE};
};

struct OceanReconstructionStats
{
    std::string method;
    std::size_t scratch_bytes{0};
    double seconds{0.};
    std::size_t selections{0};
};

// Source dimensions are FFTW (y, x); requested coordinates are padded-grid
// indices. Results are row-major over the Cartesian product (ys, xs).
class OceanReconstruction
{
public:
    OceanReconstruction(
        int source_ny,
        int source_nx,
        int ny,
        int nx,
        OceanReconstructionOptions options = {});
    ~OceanReconstruction();
    OceanReconstruction(const OceanReconstruction&) = delete;
    OceanReconstruction& operator=(const OceanReconstruction&) = delete;

    std::vector<double> evaluate(
        const std::vector<std::complex<double>>& modes,
        const std::vector<int>& xs,
        const std::vector<int>& ys,
        std::size_t reserved_bytes = 0);
    const OceanReconstructionStats& stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

// Apply the same depth coefficients as the padded HOS-Ocean velocity path,
// while keeping the original compressed source dimensions.
void ocean_velocity_modes(
    int source_ny,
    int source_nx,
    double xlen,
    double ylen,
    double depth,
    double z,
    double zsl,
    double dimL,
    double dimT,
    const std::vector<std::complex<double>>& mx,
    const std::vector<std::complex<double>>& my,
    const std::vector<std::complex<double>>& mz,
    std::vector<std::complex<double>>& u,
    std::vector<std::complex<double>>& v,
    std::vector<std::complex<double>>& w);

} // namespace modes_hosgrid
#endif
