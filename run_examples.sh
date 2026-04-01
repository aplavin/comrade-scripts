#!/bin/bash
# Example invocations of comrade_sample.jl.
#
# Parameters:
#   --npix          number of image pixels per side
#   --fov           field of view in mas (pixel size = fov/npix)
#   --fwhm          FWHM of the Gaussian convolution beam in mas (sets the mean image prior)
#   --datafile      path to VLBI visibility data (.uvf)
#   --output        path for serialized result (.jls)
#   --sigma-amp     std of log-amplitude gain prior per antenna/scan (omit for no gain fitting)
#   --sigma-phase   std of phase gain prior in radians (omit for no gain fitting)
#   --ferr          fractional error added to visibilities before fitting (default: 0)
#   --ftot          total flux prior range as lo,hi in Jy (default: 0.1,10, should encompass the expected flux of the source)
#   --quickrun      flag: fewer optimization trials + shorter MCMC for quick tests

RUN="julia --threads=auto --project comrade_sample.jl"

# Loose phase prior, with fractional error of 0.1%
$RUN --npix=128 --fov=19.2 --fwhm=8 \
    --sigma-amp=0.2 --sigma-phase=3.0 --ferr=0.001 \
    --datafile=data/U.fits \
    --output=results/U.jls

# Low frequency, large FOV, no gains fitting
$RUN --npix=140 --fov=210 --fwhm=100 \
    --datafile=data/L.uvf \
    --output=results/L.jls

# Mid frequency, with amplitude+phase gains as parameter
$RUN --npix=128 --fov=23 --fwhm=10 \
    --sigma-amp=0.2 --sigma-phase=0.2 \
    --datafile=data/C.uvf \
    --output=results/C.jls

# High frequency, small FOV, larger gain priors, custom flux range
$RUN --npix=128 --fov=8 --fwhm=4.5 \
    --sigma-amp=0.6 --sigma-phase=0.6 --ftot=0.05,5 \
    --datafile=data/Q.uvf \
    --output=results/Q.jls

# Plot all results
julia --threads=auto --project comrade_plot.jl
