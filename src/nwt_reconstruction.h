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

// HOS-NWT velocity modes are (-kx, -ky, k tanh(k)) A for the cosine potential
// modes A at z = 0, and the dphi/dt modes are dA/dt. Return the
// nondimensional time derivatives of the velocity modes for use with
// NwtReconstruction::velocity; the wavemaker rate modes are read directly.
void nwt_velocity_rate_modes(
    int source_ny,
    int source_nx,
    double xlen,
    double ylen,
    double dimL,
    const std::vector<double>& mt,
    std::vector<double>& mx_t,
    std::vector<double>& my_t,
    std::vector<double>& mz_t);

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
