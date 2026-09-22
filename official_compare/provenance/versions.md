# Official T-MAC Baseline Versions

This directory records the exact upstream revisions used for the official-vs-standalone comparison.

- T-MAC: `7042f8f73330bd083bc1e4bc5ccb3f88a4904aee`
- TVM: `36b9535ff364c484d04b384555106731049f44cd`
- llama.cpp: `eb07ecf0172230d58fff5d23a3fd6feebda35065`
- ExecuTorch: `817bd29dfbea4979a27f2f539c753bd46757ea7a`

Local official repository default location:

`~/tmac_external/T-MAC-official-x86`

The Microsoft T-MAC repository is treated as an external dependency and is not stored inside this repository.

## Relocated build note

The official T-MAC repository was originally built at:

`~/benchmark_project/T-MAC-official-x86`

It was later moved to:

`~/tmac_external/T-MAC-official-x86`

The existing TVM binaries were verified after relocation:

- `t_mac` imports from the new repository path.
- `tvm` imports from the new repository path.
- `libtvm.so` loads from the new repository path.
- LLVM is available from the new repository path.
- No broken symlinks were found.
- `libtvm.so` does not contain the old T-MAC path in its RUNPATH.

The existing `3rdparty/tvm/build` directory still contains generated CMake/Makefile references to the old absolute path.

Therefore this build directory is treated as a frozen, known-good benchmark build.

Do not use it for incremental TVM rebuilds after relocation.

If TVM must be rebuilt in the future, configure a clean build directory from the new repository location instead of manually rewriting generated CMake files.
