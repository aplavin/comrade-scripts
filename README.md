# comrade-scripts

Small scripts for running Comrade imaging on VLBI data.

## Setup

1. Install Julia 1.10
2. Start Julia in this repo:
   ```bash
   julia --project
   ```
3. In Julia, install the project dependencies:
   ```julia
   using Pkg
   Pkg.instantiate()
   ```

## Run

For exploration, it's typically faster and most convenient to run code snippets in the Julia terminal or in a notebook.
But a full end-to-end imaging script is also included here:

Main script:

```bash
julia --threads=auto --project comrade_sample.jl \
  --npix=128 --fov=23 --fwhm=10 \
  --datafile=data/C.uvf \
  --output=results/C.jls
```

Options:

- `--npix`: image size in pixels per side
- `--fov`: field of view in mas
- `--fwhm`: Gaussian prior FWHM in mas
- `--datafile`: input visibility file
- `--output`: output `.jls` file
- `--sigma-amp`: log-amplitude gain prior width; omit to disable gain fitting
- `--sigma-phase`: phase gain prior width in radians; omit to disable gain fitting
- `--ferr`: extra fractional noise to add before fitting
- `--ftot`: total flux prior range as `lo,hi` in Jy
- `--quickrun`: shorter optimization and MCMC run

See `run_examples.sh` for a few ready-made commands.

To make figures from `results/*.jls`:

```bash
julia --project comrade_plot.jl
```
