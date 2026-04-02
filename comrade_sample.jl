using ArgParse
using Distributions: Uniform

parse_ftot(ftot_str::AbstractString) = begin
    ftot_vals = parse.(Float64, split(ftot_str, ","))
    length(ftot_vals) == 2 || error("--ftot expects two comma-separated values, e.g. --ftot=0.1,10")
    Uniform(ftot_vals...)
end

function parse_args(args)
    settings = ArgParseSettings(
        description="Run Comrade imaging on VLBI data.",
        autofix_names=true,
    )
    @add_arg_table! settings begin
        "--npix"
            help = "image size in pixels per side"
            arg_type = Int
            required = true
        "--fov"
            help = "field of view in mas"
            arg_type = Float64
            required = true
        "--fwhm"
            help = "Gaussian prior FWHM in mas"
            arg_type = Float64
            required = true
        "--datafile"
            help = "input visibility file"
            arg_type = String
            required = true
        "--output"
            help = "output .jls file"
            arg_type = String
            required = true
        "--sigma-amp"
            help = "log-amplitude gain prior width; provide together with --sigma-phase"
            arg_type = Float64
        "--sigma-phase"
            help = "phase gain prior width in radians; provide together with --sigma-amp"
            arg_type = Float64
        "--ferr"
            help = "extra fractional noise to add before fitting"
            arg_type = Float64
            default = 0.0
        "--ftot"
            help = "total flux prior range as lo,hi in Jy"
            arg_type = String
            metavar = "LO,HI"
            default = "0.1,10"
        "--quickrun"
            help = "fewer optimization trials and shorter MCMC for quick tests"
            action = :store_true
    end
    parsed = ArgParse.parse_args(args, settings; as_symbols=true)
    isnothing(parsed[:sigma_amp]) == isnothing(parsed[:sigma_phase]) || error("--sigma-amp and --sigma-phase must be provided together")
    return (
        npix=parsed[:npix],
        fov=parsed[:fov],
        fwhm=parsed[:fwhm],
        datafile=parsed[:datafile],
        σamp=parsed[:sigma_amp],
        σphase=parsed[:sigma_phase],
        quickrun=parsed[:quickrun],
        output=parsed[:output],
        ferr=parsed[:ferr],
        ftot=parse_ftot(parsed[:ftot]),
    )
end

cli = parse_args(ARGS)

using VLBIFiles
using Comrade
using VLBISkyModels, VLBIImagePriors, Distributions
using Optimization, OptimizationLBFGSB, StableRNGs
using Enzyme
using AdvancedHMC
using DataManipulation
using Accessors
using Unitful, UnitfulAngles
using LinearAlgebra, Random, Serialization, Statistics

# for logging to flush immediately
using LoggingExtras
global_logger(MinLevelLogger(FileLogger("/dev/stderr"), Logging.Info))

include("comrade_schedule.jl")
include("comrade_setup.jl")

fwhmfac = 2 * sqrt(2 * log(2))

npix = cli.npix
fovxy = μas2rad(cli.fov * 1000)
grid = imagepixels(fovxy, fovxy, npix, npix)

recipe = ImagingRecipe(
    TemperedSchedule(
        stages=[
            OptStage(noise_frac=0.05, maxiters=1000, topk=10),
            OptStage(noise_frac=0.03, maxiters=1000, topk=5),
            OptStage(noise_frac=0.01, maxiters=1000, topk=3),
            OptStage(noise_frac=0.0, maxiters=1000, topk=1),
        ],
        ntrials=cli.quickrun ? 3 : 30,
        optimizer=LBFGSB(),
    ),
    SamplingConfig(sampler=NUTS(0.8), nsample=cli.quickrun ? 50 : 300, nadapt=cli.quickrun ? 25 : 2000, thinning=cli.quickrun ? 1 : 15),
    cli.quickrun ? 25 : 50,
)

fwhm_μas = cli.fwhm * 1000.0
fwhm_str = isinteger(cli.fwhm) ? string(Int(cli.fwhm)) : string(cli.fwhm)
label = isnothing(cli.σamp) ? "FWHM=$(fwhm_str)mas, no gains" : "FWHM=$(fwhm_str)mas, w/gains $(cli.σamp)"
cfg = (label=label, fwhm=fwhm_μas, order=1, σimg_prior=Exponential(1.0), σamp=cli.σamp, σphase=cli.σphase)

datafile = cli.datafile

mkpath("./results")

(; uvtbl_orig, uvtbl_I_avg, cmrd_data) = load_uvdata(datafile)
if cli.ferr > 0
    cmrd_data = add_fractional_noise(cmrd_data, cli.ferr)
end
@info "Processing $datafile, config: $(cfg.label)"
mpr = VLBISkyModels.modify(Gaussian(), Stretch(μas2rad(cfg.fwhm) ./ fwhmfac))
imgpr = intensitymap(mpr, grid)
skymeta = (; mimg=imgpr ./ Comrade.flux(imgpr))

cprior = corr_image_prior(grid, cmrd_data; cfg.order)
prior = (c = cprior, σimg = cfg.σimg_prior, ftot = cli.ftot)
skym = SkyModel(sky, prior, grid; metadata=skymeta)

rng = StableRNG(42)
intm = build_instrument(; cfg.σamp, cfg.σphase)
result = comrade_imager(skym, intm, cmrd_data; recipe, rng)

outpath = cli.output
open(outpath, "w") do io
    serialize(io, Dict(
        "datafile" => datafile,
        "config" => cfg,
        "npix" => npix,
        "fovxy" => fovxy,
        "result" => result,
        "uvtbl_data" => uvtbl_I_avg,
    ))
end
println("$(cfg.label) → saved to $outpath")
