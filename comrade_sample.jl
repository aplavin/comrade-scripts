using Distributions: Uniform

# --- CLI arguments ---
function parse_args(args)
    parsed = Dict{String,String}()
    for arg in args
        if startswith(arg, "--")
            parts = split(arg[3:end], "=", limit=2)
            parsed[parts[1]] = length(parts) == 2 ? parts[2] : ""
        else
            error("Unknown argument format: $arg. Use --key=value.")
        end
    end
    npix = parse(Int, parsed["npix"])
    fov = parse(Float64, parsed["fov"])  # in mas
    fwhm = parse(Float64, parsed["fwhm"])  # in mas
    datafile = parsed["datafile"]
    σamp = haskey(parsed, "sigma-amp") ? parse(Float64, parsed["sigma-amp"]) : nothing
    σphase = haskey(parsed, "sigma-phase") ? parse(Float64, parsed["sigma-phase"]) : nothing
    quickrun = haskey(parsed, "quickrun")
    output = parsed["output"]
    ferr = haskey(parsed, "ferr") ? parse(Float64, parsed["ferr"]) : 0.0
    ftot_str = get(parsed, "ftot", "0.1,10")
    ftot_vals = parse.(Float64, split(ftot_str, ","))
    length(ftot_vals) == 2 || error("--ftot expects two comma-separated values, e.g. --ftot=0.1,10")
    ftot = Uniform(ftot_vals...)
    (; npix, fov, fwhm, datafile, σamp, σphase, quickrun, output, ferr, ftot)
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
