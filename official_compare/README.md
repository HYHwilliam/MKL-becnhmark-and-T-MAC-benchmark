# Official T-MAC Comparison

This directory contains the files used to compare the standalone x86 implementation in this repository with Microsoft T-MAC.

## Structure

- `adapters/`: local benchmark adapter used with the official T-MAC implementation.
- `scripts/`: benchmark runners and environment setup.
- `provenance/`: exact upstream revisions and environment notes.
- `results/summary/`: curated results intended for Git.
- `results/raw/`: preserved raw results, ignored by Git.
- `results/runs/`: newly generated benchmark runs, ignored by Git.

## Official T-MAC

The Microsoft T-MAC repository is kept outside this repository at:

`~/tmac_external/T-MAC-official-x86`

Pinned T-MAC commit:

`7042f8f73330bd083bc1e4bc5ccb3f88a4904aee`

See `provenance/versions.md` for the complete dependency revisions.

## Environment

Load the comparison environment with:

`source official_compare/scripts/tmac_env.sh`

The environment uses the existing `tvm-build` Conda environment and the T-MAC/TVM build in the external repository.

The relocated TVM build is treated as a frozen benchmark build. The existing TVM build directory should not be used for incremental rebuilding after relocation.

## Benchmark Scripts

Combined standalone VLA and official HackMD-safe benchmark:

`official_compare/scripts/run_vla_and_hackmd_compare.sh`

Official VLA and HackMD benchmark:

`official_compare/scripts/run_official_overnight.sh`

Each new run is written to a separate directory under:

`official_compare/results/runs/`

Existing runs are not automatically deleted.
