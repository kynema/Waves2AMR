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
grid area, and velocity storage also scales with height count.

Build and run the focused tests:

```sh
cmake -S . -B build -DFFTW_DIR=/path/to/fftw
cmake --build build --target Waves2AMR_padding_tests -j4
./build/unit_tests/Waves2AMR_padding_tests
```
