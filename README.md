# Waves2AMR
for bringing HOS waves into AMR-Wind simulations

## Resolution-aware HOS-Ocean reconstruction

`modes_hosgrid::ocean_fft_dimensions(ny, nx, ly, lx, dy, dx, cap)` selects
inverse-FFT dimensions by independently doubling each source dimension until
its physical spacing is no larger than the target spacing. Dimensions use
FFTW order (full y, compressed x); lengths and spacings use physical units.
The default cap is `-1` (uncapped); `0` disables padding, and positive `p`
allows at most `2^p` expansion per direction. No dimension is reduced.

For the original full-grid API, after each mode read use
`pad_ocean_modes(ny, nx, padded_ny, padded_nx, modes)`
for elevation and all three velocity mode vectors, before their existing
reconstruction routines. Allocate FFT buffers/plans and spatial arrays using
the padded dimensions, and interpolate using the correspondingly smaller
spacings. Retain the original reader dimensions for subsequent reads.
The helper preserves HOS normalization, relocates negative frequencies,
splits even-grid Nyquist modes, and leaves newly introduced modes zero.
It does not add wave content. HOS-NWT uses the endpoint-aware path below,
not `pad_ocean_modes`.

Padding reduces subsequent interpolation loss but does not eliminate
interpolation or levelset/VOF reconstruction error. Memory scales with padded
grid area.

## Cubic interpolation

The elevation and velocity interpolation APIs default to
`interp_to_mfab::Interpolation::cubic` for both HOS-Ocean and HOS-NWT.
They interpolate reconstructed Fourier-grid values, not the Fourier series
directly at destination coordinates. Horizontal interpolation is tensor-product
four-point Lagrange (bicubic for 3D sources); velocity uses four adjacent,
potentially nonuniform heights for cubic vertical interpolation.
Pass `Interpolation::linear` as the final argument for legacy interpolation.
Kynema-SGF exposes this choice as `OceanWaves.<label>.interpolation = cubic|linear`.

Periodic horizontal stencils wrap; nonperiodic coordinates clamp to the actual
sample extent `[0, (n-1)*spacing]` and use one-sided stencils at boundaries.
This endpoint policy intentionally differs from legacy linear interpolation.
Axes with fewer than four samples reduce polynomial degree. Vertical stencils
are one-sided near endpoints; outside the supplied height range, velocity
retains linear extrapolation and the existing above-top zeroing rule.
These are interpolating polynomials, not monotonicity-limited schemes:
overshoot is possible, particularly near nonsmooth data.

## Compact local reconstruction

`modes_hosgrid::OceanReconstruction(source_ny, source_nx, ny, nx, options)`
keeps source modes compact and evaluates the same padded-grid nodes without
materializing padded mode vectors. Call `evaluate(modes, xs, ys)` with original
compressed HOS coefficients and sorted, unique padded-grid indices. The return
value is row-major over the Cartesian product `(ys, xs)`. Source coefficients
are not modified. Normalization and Nyquist splitting match `pad_ocean_modes`
followed by the original inverse FFT, to floating-point roundoff.

`OceanReconstructionOptions` selects `auto` (default), `full`, or `pruned`,
with a default accounted scratch budget of 512 MiB. `full` uses a full FFTW
complex-to-real transform and gathers the requested nodes. `pruned` uses
separable complex FFTs: transform occupied frequency rows, retain only requested
coordinates, then transform only those coordinates in the other direction.
Both x-first and y-first orders are available. This is input/output pruning
at the row/column level, not pruning inside FFTW's individual 1D transforms.

Automatic selection rejects methods exceeding the budget, filters candidates
whose estimated transform work is more than four times the cheapest feasible
candidate, then warms each remaining candidate and times two executions.
A later candidate must be at least 10% faster to replace the current best.
Plans and the decision are cached until requested indices or reserved scratch
change. Calibration excludes planning and compact-mode preparation; normal
`evaluate` timing includes compact-mode preparation. Existing FFTW planner flags
are supported. Selection is rank-local and requires no MPI communication.

The reported scratch estimate counts the compact expanded spectrum, transform
buffers/intermediates, one returned host array and its device copy, plus the
optional `reserved_bytes` argument to `evaluate`. It excludes source input
arrays, FFTW plan internals, allocator overhead, and destination mesh fields;
it is not a hard cap on process RSS. If no permitted method fits, reconstruction
fails explicitly. An empty request performs no FFT.

`ocean_velocity_modes` computes compact depth-dependent velocity coefficients
using the same depth-factor helper as the original velocity reconstruction.
It preserves that routine's existing first/second-dimension length convention.

### HOS-NWT reconstruction

`nwt_fft_dimensions(ny, nx, ly, lx, dy, dx, cap)` applies the same padding
controls to NWT. Its grids include both tank endpoints: doubling the number
of intervals changes `n` samples to `2*(n-1)+1`, not `2*n`. A single transverse
row remains one row, irrespective of SGF's transverse resolution.

`NwtReconstruction(source_ny, source_nx, ny, nx, options)` accepts the original
real NWT coefficients. `evaluate(modes, cos_y, cos_x, xs, ys, reserved_bytes)`
returns a row-major Cartesian product of requested endpoint-grid nodes.
It represents the cosine/sine basis through even/odd reflection and reuses
the compact full/pruned/automatic FFT backend. Endpoint cosine amplitudes are
preserved when padding; endpoint sine modes remain absent, as in DST-I.
The internal periodic extension is `2*(n-1)` per nontrivial axis, so a full
transform can use more workspace than a native real-to-real transform.
Automatic selection times these available methods; it does not compare
against the legacy DCT/DST implementation. No direct Fourier evaluation at
solver coordinates is introduced.

`velocity` applies the original NWT depth factors and dimensionalization,
including the NWT coefficient cap of 3 (not Ocean's different caps). It also
retains the additional wavemaker modes. Their nonperiodic x dependence is
evaluated only at requested reconstruction-grid x nodes; their y dependence
uses padded cosine/sine FFTs, one x column at a time. These terms cannot be
zero-padded as if they were periodic x modes. They use full 1D y transforms
even when `pruned` is selected for the ordinary wave modes.

The shared scratch budget includes the reflected compact coefficients,
depth arrays, local returned samples and wavemaker FFT buffers, plus caller
reservations. The same exclusions and rank-local method selection apply.
Inputs and reader dimensions remain unchanged. With padding disabled, results
match the legacy transforms to roundoff, not necessarily bitwise.

## Streaming velocity reconstruction

`interp_to_mfab::stream_velocity_to_field` reconstructs velocities directly into
locally owned MultiFab boxes, including ghosts, across the supplied AMR levels.
Pass the same descending height vector and contiguous height indices as the
existing interpolator. The callback receives a height index and three
single-plane device vectors; fill them with `populate_hos_vel` at that height,
using the default zero spatial offset. Geometry arrays must contain one entry
per supplied field. Include cubic support heights in the contiguous index range:
expand the legacy bracketing range by two heights at each end, clipped to the
height vector. Fewer supplied heights reduce order.

The original callback reconstructs only heights needed by local boxes. It holds
three full horizontal velocity planes. Cubic interpolation immediately reduces
each plane to local horizontal samples and retains four heights for each of
three components. Linear mode retains two sets of four corners per component
to preserve its arithmetic. Spatial scratch storage is therefore
`sizeof(Real) * (3 * nx * ny + s * local_horizontal_cells)`, where `s` is
12 for cubic (with four available heights), or 24 for linear, independent of the
total number of heights. Here `local_horizontal_cells` sums grown-box horizontal
areas across local boxes and levels, including vertically stacked boxes.
For small source grids or many stacked boxes, local stencil storage can exceed
the old volume storage; the benefit grows with padded FFT area and height count.
The returned `VelocityStreamMemory` reports plane and stencil bytes, excluding
FFT plans/workspace, spectral vectors, elevation, and destination fields.
An empty local mesh performs no reconstructions.

Alternatively, pass an empty original callback and a `LocalVelocityPlaneReader`
as the local-reader argument, before the optional interpolation selector.
This callback receives the height and sorted unique
`xs, ys` needed by all local boxes, including ghosts and periodic seams, and
fills three arrays over their Cartesian product. `local_eta_to_levelset_field`
provides the analogous elevation path. Both paths use the same horizontal
stencil calculation and interpolation selector.

Kynema-SGF uses compact local reconstruction for both HOS-Ocean and HOS-NWT elevation and streamed
velocity, including initialization after boxes exist. Its budget also accounts
for the three local velocity arrays, local stencils, device index maps and
compact depth coefficients. Initialization processes each level separately.
Compact input modes remain replicated per rank; no distributed FFT is used.
Kynema-SGF reads headers and retained coefficients only on its I/O rank, then
broadcasts metadata and compact coefficients to all ranks. EOF status and loop
offsets are synchronized before coefficients; averaging and ghost exchanges
are unchanged. A repeated request for the already-distributed timestep skips
both parsing and coefficient broadcasts.
Fragmented boxes may have a Cartesian-product footprint larger than the exact
union of requested points. Cubic support can require more FFT nodes and heights
than linear, so smaller local stencils do not guarantee lower total memory or
runtime. Both source types defer spatial reconstruction until boxes exist and
avoid storing full horizontal planes at every interpolation height.

### Modes I/O

Both ASCII readers retain a binary file handle and seek directly to HOS
SWENSE direct-access records (18-byte scalar slots). Ocean records contain
12 scalar slots per complex-mode location; NWT records reserve eight slots
per ordinary-mode location, with each additional-mode block padded to the
ordinary x width. The special
header/initialization record at timestep `-1` retains its original layout.
Files must use this fixed-width format and must not be modified while open.
Retained blocks are parsed with finite-value checks; brief reads skip the
time-derivative blocks without numeric conversion, except the displaced NWT
initialization record, which retains legacy sequential parsing.

Legacy `get_data`/`read_data` APIs keep a last-successful-record cache.
`get_data` leaves caller vectors untouched on EOF or a parsing error.
EOF returns `false`; malformed headers, retained values, partial records,
invalid indices and I/O failures throw exceptions. Reader copies share the
file handle but keep independent coefficient caches; sequential use is
supported, concurrent reads on a reader or its copies are not.

For applications that own their coefficient arrays, initialize with
`initialize(filename, is_ocean, false, false)` and use
`read_into(step, x, y, z, fs)` (Ocean) or
`read_into(step, x, y, z, fs, &additional)` (NWT). This avoids internal
coefficient storage and output copies. The direct API resizes destinations,
does not cache, and may partially modify them before throwing a parsing error;
do not use failed results. EOF leaves them untouched. `records_read()` counts
successful parses, excluding legacy cache hits. `metadata()` exports
dimensional metadata; `initialize(metadata, is_ocean)` creates a metadata-only
reader on ranks that do not access the file.

Build and run the focused tests:

```sh
cmake -S . -B build -DFFTW_DIR=/path/to/fftw
cmake --build build --target Waves2AMR_padding_tests -j4
./build/unit_tests/Waves2AMR_padding_tests
```

Measure reconstruction speed and accounted memory separately from solver costs:

```sh
cmake --build build --target Waves2AMR_reconstruction_benchmark -j4
# source width, padded width, local square width, timed repeats
./build/tests/Waves2AMR_reconstruction_benchmark 256 4096 512 10
```

The benchmark compares all choices with the full FFT and fails if the maximum
absolute discrepancy exceeds `1e-10`. It reports steady-state time per scalar
plane, excluding first-call selection and planning. On an Apple M4, 256-square
source modes on a 4096-square output grid gave roughly 13.6x and 6.1x faster
reconstruction for 256-square and 1024-square local footprints, respectively.
Accounted scratch decreased from 258 to 3.1 MiB and from 273 to 21.1 MiB.
A full 128-square footprint with 64-square source modes selected `full`.
These are illustrative host-side measurements, not guaranteed solver speedups.
