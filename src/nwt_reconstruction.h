#ifndef NWT_RECONSTRUCTION_H
#define NWT_RECONSTRUCTION_H

#include "ocean_reconstruction.h"
#include <array>

namespace modes_hosgrid {

// NWT dimensions include both endpoints. A single transverse row stays 2D.
std::array<int, 2> nwt_fft_dimensions(
    int ny, int nx, double ly, double lx, double dy, double dx, int cap = -1);

class NwtReconstruction
{
public:
    NwtReconstruction(
        int source_ny,
        int source_nx,
        int ny,
        int nx,
        OceanReconstructionOptions options = {});
    ~NwtReconstruction();
    NwtReconstruction(const NwtReconstruction&) = delete;
    NwtReconstruction& operator=(const NwtReconstruction&) = delete;

    std::vector<double> evaluate(
        const std::vector<double>& modes,
        bool cos_y,
        bool cos_x,
        const std::vector<int>& xs,
        const std::vector<int>& ys,
        std::size_t reserved_bytes = 0);

    std::array<std::vector<double>, 3> velocity(
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
        std::size_t reserved_bytes = 0);
    const OceanReconstructionStats& stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

std::pair<double, double>
nwt_depth_factors(double k, double z, bool first_column, bool zero_mode);

// Wavemaker terms are nonperiodic in x, but cosine/sine modes in y.
void nwt_additional_coefficients(
    int ny,
    int count,
    double xlen,
    double ylen,
    double x,
    double z,
    const std::vector<double>& modes,
    double* u,
    double* v,
    double* w);

} // namespace modes_hosgrid
#endif
