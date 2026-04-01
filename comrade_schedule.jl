
# ---------- tempered optimization types ----------

@kwdef struct OptStage
    noise_frac::Float64
    maxiters::Int
    topk::Int
    g_tol::Float64 = 1e-1
end

struct OptCandidate{T}
    params::T
    logp::Float64
end

@kwdef struct TemperedSchedule
    stages::Vector{OptStage}
    ntrials::Int
    optimizer
end

# ---------- sampling / recipe / result types ----------

@kwdef struct SamplingConfig
    sampler = NUTS(0.8)
    nsample::Int
    nadapt::Int
    thinning::Int = 1
end

struct ImagingRecipe
    optimization::TemperedSchedule
    sampling::Union{SamplingConfig, Nothing}
    nimgs::Int
end

struct ImagingResult
    stage_xopts::Vector   # best params after each optimization stage
    xopt                  # final optimized params
    post                  # posterior (clean, no noise)
    samples::Vector       # selected post-warmup posterior samples
end

# ---------- helpers ----------

function _make_posterior(skym, intm, data)
    args = isnothing(intm) ? (skym, data) : (skym, intm, data)
    return VLBIPosterior(args...; admode=set_runtime_activity(Enzyme.Reverse))
end

# ---------- tempered optimization functions ----------

function rank_and_trim(candidates, topk)
    valid = filter(c -> !isnan(c.logp), candidates)
    sort!(valid; by=c -> c.logp, rev=true)
    return valid[1:min(topk, length(valid))]
end

function optimize_stage(post, initial_params, stage::OptStage, optimizer)
    candidates = map(enumerate(initial_params)) do (i, x0)
        xopt, sol = comrade_opt(post, optimizer; initial_params=x0, maxiters=stage.maxiters, g_tol=stage.g_tol)
        logp = sum(logdensityof(post, xopt))
        @info "Optimization $i/$(length(initial_params)): logp = $logp"
        OptCandidate(xopt, logp)
    end
    return rank_and_trim(candidates, stage.topk)
end

function tempered_optimize(skym, intm, data, schedule::TemperedSchedule;
                           rng=Random.default_rng())
    candidates = nothing
    stage_xopts = []

    for (i, stage) in enumerate(schedule.stages)
        data_i = stage.noise_frac == 0 ? data : add_fractional_noise(data, stage.noise_frac)
        post_i = _make_posterior(skym, intm, data_i)

        if isnothing(candidates)
            candidates = [(;params=prior_sample(rng, post_i)) for _ in 1:schedule.ntrials]
        end
        candidates = optimize_stage(
            post_i,
            [c.params for c in candidates],
            stage, schedule.optimizer)

        push!(stage_xopts, candidates[1].params)
        @info "Stage $i/$(length(schedule.stages)) complete" noise=stage.noise_frac best_logp=candidates[1].logp
    end

    return stage_xopts
end

# ---------- main entry point ----------

function comrade_imager(skym, intm, data; recipe, rng=Random.default_rng())
    (; optimization, sampling, nimgs) = recipe

    stage_xopts = tempered_optimize(skym, intm, data, optimization; rng)
    xopt = last(stage_xopts)
    post = _make_posterior(skym, intm, data)

    if isnothing(sampling)
        return ImagingResult(stage_xopts, xopt, post, [])
    end

    (; sampler, nsample, nadapt, thinning) = sampling
    chain = sample(rng, post, sampler, nsample; initial_params=xopt, n_adapts=nadapt, thinning)

    all_samples = Comrade.postsamples(chain)
    warmup_mask = [!s.is_adapt for s in Comrade.samplerstats(chain)]
    post_samples = all_samples[warmup_mask]
    ixs = round.(Int, range(1, length(post_samples), length=min(nimgs, length(post_samples))))
    selected = post_samples[ixs]

    return ImagingResult(stage_xopts, xopt, post, selected)
end
