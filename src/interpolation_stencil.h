#ifndef INTERPOLATION_STENCIL_H
#define INTERPOLATION_STENCIL_H

#include "AMReX_Gpu.H"
#include <cmath>

namespace interp_to_mfab {

enum class Interpolation { linear, cubic };

struct PolynomialStencil
{
    int count{0};
    int index[4]{};
    amrex::Real weight[4]{};
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE PolynomialStencil lagrange_stencil(
    amrex::Real coordinate, const amrex::Real* nodes, int first, int count)
{
    PolynomialStencil result;
    result.count = count;
    for (int a = 0; a < count; ++a) {
        result.index[a] = first + a;
        result.weight[a] = 1.;
        for (int b = 0; b < count; ++b) {
            if (a != b)
                result.weight[a] *=
                    (coordinate - nodes[b]) / (nodes[a] - nodes[b]);
        }
    }
    return result;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE PolynomialStencil horizontal_cubic(
    amrex::Real coordinate, int size, amrex::Real spacing, bool periodic)
{
    if (size == 1) {
        PolynomialStencil result;
        result.count = 1;
        result.weight[0] = 1.;
        return result;
    }
    auto x = coordinate / spacing;
    if (periodic) {
        x -= std::floor(x / size) * size;
    } else {
        x = amrex::min(amrex::Real(size - 1), amrex::max(amrex::Real(0), x));
    }
    const int count = amrex::min(4, size);
    int first = static_cast<int>(std::floor(x)) - (count - 1) / 2;
    if (!periodic) first = amrex::max(0, amrex::min(size - count, first));
    amrex::Real nodes[4];
    for (int n = 0; n < count; ++n) nodes[n] = first + n;
    auto result = lagrange_stencil(x, nodes, first, count);
    for (int n = 0; n < count; ++n) {
        if (periodic) result.index[n] = (result.index[n] % size + size) % size;
    }
    return result;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
vertical_stencil_start(int upper, int first, int last)
{
    const int count = amrex::min(4, last - first + 1);
    return amrex::max(first, amrex::min(last - count + 1, upper - 1));
}

} // namespace interp_to_mfab
#endif
