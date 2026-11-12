# Waves2AMR
for bringing HOS waves into AMR-Wind simulations

## Resolution-aware HOS-Ocean reconstruction

`modes_hosgrid::ocean_fft_dimensions(ny, nx, ly, lx, dy, dx, cap)` selects
inverse-FFT dimensions by independently doubling each source dimension until
its physical spacing is no larger than the target spacing. Dimensions use
FFTW order (full y, compressed x); lengths and spacings use physical units.
The default cap is `-1` (uncapped); `0` disables padding, and positive `p`
allows at most `2^p` expansion per direction. No dimension is reduced.

After each mode read, use `pad_ocean_modes(ny, nx, padded_ny, padded_nx, modes)`
for elevation and all three velocity mode vectors, before their existing
reconstruction routines. Allocate FFT buffers/plans and spatial arrays using
the padded dimensions, and interpolate using the correspondingly smaller
spacings. Retain the original reader dimensions for subsequent reads.
The helper preserves HOS normalization, relocates negative frequencies,
splits even-grid Nyquist modes, and leaves newly introduced modes zero.
It does not add wave content. HOS-NWT transforms are unchanged.

Padding reduces subsequent linear interpolation loss but does not eliminate
interpolation or levelset/VOF reconstruction error. Memory scales with padded
grid area.

## Streaming velocity reconstruction

`interp_to_mfab::stream_velocity_to_field` reconstructs velocities directly into
locally owned MultiFab boxes, including ghosts, across the supplied AMR levels.
Pass the same descending height vector and contiguous height indices as the
existing interpolator. The callback receives a height index and three
single-plane device vectors; fill them with `populate_hos_vel` at that height,
using the default zero spatial offset. Geometry arrays must contain one entry
per supplied field. Periodic and nonperiodic interpolation, including
single-row (2D) source grids, follow the existing interpolation arithmetic.

The routine reconstructs only heights needed by local boxes. It holds three
full horizontal velocity planes and two local sets of four-corner samples for
each velocity component. Spatial scratch storage is therefore
`sizeof(Real) * (3 * nx * ny + 24 * local_horizontal_cells)`, independent of the
number of heights. Here `local_horizontal_cells` sums grown-box horizontal
areas across local boxes and levels, including vertically stacked boxes.
For small source grids or many stacked boxes, local stencil storage can exceed
the old volume storage; the benefit grows with padded FFT area and height count.
The returned `VelocityStreamMemory` reports plane and stencil bytes, excluding
FFT plans/workspace, spectral vectors, elevation, and destination fields.
An empty local mesh performs no reconstructions.

Kynema-SGF uses this path for HOS-Ocean initialization (after local boxes exist)
and mode updates. Initialization processes each AMR level separately, which
can repeat FFTs in exchange for lower memory use. Full horizontal FFTs still
run on each rank; this is not a distributed or pruned FFT. HOS-NWT retains its
existing reconstruction path.

Build and run the focused tests:

```sh
cmake -S . -B build -DFFTW_DIR=/path/to/fftw
cmake --build build --target Waves2AMR_padding_tests -j4
./build/unit_tests/Waves2AMR_padding_tests
```
