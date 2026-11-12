# Waves2AMR
for bringing HOS waves into AMR-Wind simulations, written generally for AMReX-based codes

## Modes I/O

Ocean and NWT readers retain an open handle and seek directly to fixed-width
SWENSE records. Brief reads skip derivatives; legacy reads cache the last
successful timestep. EOF returns `false`, while malformed data, partial records,
invalid indices and I/O failures throw. Files must not change while open.
Reader copies support sequential, not concurrent, use of the shared handle.

`initialize(filename, is_ocean, false, false)` with `read_into(step, x, y, z, fs)`
(Ocean) or `read_into(step, x, y, z, fs, &additional)` (NWT) avoids internal
coefficient storage and output copies. Direct reads resize destinations, have
no cache, and may partially modify them before throwing; EOF leaves them
untouched. `metadata()` and `initialize(metadata, is_ocean)` support metadata-only
readers on non-I/O ranks. Kynema-SGF reads only on its I/O rank, broadcasts compact
modes to all ranks and caches the distributed timestep. Its field exchanges
are unchanged.
