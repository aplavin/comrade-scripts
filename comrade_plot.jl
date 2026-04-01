using Serialization, Statistics
using VLBIFiles, VLBIPlots, Comrade, VLBISkyModels, VLBIImagePriors
using Distributions, Optimization, OptimizationLBFGSB, Enzyme, AdvancedHMC
using Uncertain
using MakieExtra; import GLMakie
using DataManipulation, Accessors, AxisKeysExtra
using Unitful, UnitfulAngles
using LinearAlgebra, StableRNGs, Random
using OhMyThreads: tmap

include("comrade_schedule.jl")
include("comrade_setup.jl")

using VLBISkyModels: ModifiedModel, AddModel, ConvolvedModel

# Override intensity_point to only loop over pixels within radialextent of the pulse,
# instead of all pixels. Changes scaling from O(npix⁴) to O(npix² × radialextent²).
function VLBISkyModels.intensity_point(m::ContinuousImage, p)
    dx, dy = pixelsizes(m.img)
    re = VLBISkyModels.radialextent(m.kernel)
    g = axisdims(m.img)
    xs, ys = g.X, g.Y
    rx = re * abs(dx)
    ry = re * abs(dy)

    ixrng = searchsortedfirst(xs, p.X - rx):searchsortedlast(xs, p.X + rx)
    iyrng = searchsortedfirst(ys, p.Y - ry):searchsortedlast(ys, p.Y + ry)

    sum = zero(eltype(m.img))
    ms = stretched(m.kernel, dx, dy)
    @inbounds for I in CartesianIndices((ixrng, iyrng))
        ix, iy = Tuple(I)
        dp = (X=(p.X - xs[ix]), Y=(p.Y - ys[iy]))
        sum += m.img[I] * VLBISkyModels.intensity_point(ms, dp)
    end
    return sum
end

result_files = filter(endswith(".jls"), readdir("./results"; join=true))
if !isempty(ARGS)
    result_files = filter(f -> occursin(ARGS[1], f), result_files)
end

mkpath("./figs")

for rfile in result_files
    slug = chopsuffix(basename(rfile), ".jls")
    d = open(deserialize, rfile)
    (; post, stage_xopts, xopt, samples) = d["result"]
    cfg = d["config"]
    uvtbl_I_avg = d["uvtbl_data"]
    grid = imagepixels(d["fovxy"], d["fovxy"], d["npix"], d["npix"])

    # --- main figure ---
    fig = Figure()
    update_theme!(Axis=(width=350, height=350))

    fplt_amp = RadPlot(nothing, markersize=3, axis=(;height=260))
    fplt_phas = RadPlot(nothing, yfunc=angle, markersize=3, axis=(;height=110))

    fplt_resid = FPlot(nothing,
        VLBIPlots.F.UVdist(), AxFunc(x->clamp(x.value, 0, 10), label="Residual (nσ)", limit=(0, 10)),
        markersize=5)

    # --- optimization stage columns ---
    for (col, sx) in enumerate(stage_xopts[end:end])  # can show multiple optimization stages if desired; now only showing latest
        img = @p intensitymap(skymodel(post, sx), grid) |> KeyedArray |> @modify(ak -> ak .|> u"mas", axiskeys(__)[∗])
        uvtbl_m = VLBI.uvtable(simulate_observation(post, sx; add_thermal_noise=false)[1])

        ax_img = Axis(fig[1, col], title="Stage $col", aspect=1)
        image!(ax_img, img, colormap=:afmhot, colorscale=SymLog(1e-2 * maximum(img)))

        axplot(scatter)(fig[2, col][1,1], merge(fplt_amp, FPlot(uvtbl_I_avg, color=:black)), label="Data")
        scatter!(merge(fplt_amp, FPlot(uvtbl_m, color=:red)), label="Model")
        col == 1 && axislegend()

        axplot(scatter)(fig[2, col][2,1], merge(fplt_phas, FPlot(uvtbl_I_avg; color=:black)), label="Data")
        scatter!(merge(fplt_phas, FPlot(uvtbl_m; color=:red)), label="Model")

        res_vals = abs.(U.value.(uvtbl_I_avg.value - uvtbl_m.value)) ./ U.uncertainty.(uvtbl_I_avg.value)
        uvtbl_res = @set uvtbl_m.value = res_vals
        axplot(scatter)(fig[2, col][3,1], (@set fplt_resid.data = uvtbl_res), axis=(;height=110))
    end

    # --- MCMC posterior columns ---
    imgs_data = map(s -> KeyedArray(intensitymap(skymodel(post, s), grid)), samples)
    img_mean = mean(imgs_data)
    img_std = std(imgs_data)
    img_relstd = img_std ./ (img_mean .+ eps(Float64))

    # posterior mean column
    uvtbls_sim = map(s -> VLBI.uvtable(simulate_observation(post, s; add_thermal_noise=false)[1]), samples)
    uvtbl_mean = @set $(uvtbls_sim[1]).value = mean(u -> u.value, uvtbls_sim)

    ax_img = Axis(fig[1, end+1], title="MCMC mean", aspect=1)
    image!(ax_img, img_mean, colormap=:afmhot, colorscale=SymLog(1e-2 * maximum(img_mean)))
    
    axplot(scatter)(fig[2, end][1,1], merge(fplt_amp, FPlot(uvtbl_I_avg, color=:black)), label="Data")
    scatter!(merge(fplt_amp, FPlot(uvtbl_mean, color=:red)), label="Model")
    axplot(scatter)(fig[2, end][2,1], merge(fplt_phas, FPlot(uvtbl_I_avg; color=:black)), label="Data")
    scatter!(merge(fplt_phas, FPlot(uvtbl_mean; color=:red)), label="Model")

    res_vals = abs.(U.value.(uvtbl_I_avg.value - uvtbl_mean.value)) ./ U.uncertainty.(uvtbl_I_avg.value)
    uvtbl_res = @set uvtbl_mean.value = res_vals
    axplot(scatter)(fig[2, end][3,1], (@set fplt_resid.data = uvtbl_res), axis=(;height=110))

    # uncertainty column: stdev (row 1), stdev/mean (row 2)
    ax_std = Axis(fig[1, end+1], title="Posterior σ", aspect=1)
    image!(ax_std, img_std, colormap=:viridis)
    ax_snr = Axis(fig[2, end], title="σ / mean", aspect=1)
    image!(ax_snr, img_relstd, colormap=:viridis, colorrange=(0, 1))

    Label(fig[0, :], cfg.label, fontsize=18)

    resize_to_layout!()

    save("./figs/$slug.png", fig)
    println("$(cfg.label) → saved to figs/$slug.png")

    # --- gain plot (only for models with gains) ---
    if !isnothing(cfg.σamp) && !isempty(samples)
        # extract gain arrays from all samples, centering per-sample
        # (same centering as forward_model applies during sampling)
        all_lg = [let v = parent(s.instrument.lg); v .- mean(v) end for s in samples]
        all_gp = [let v = parent(s.instrument.gp); v .- mean(v) end for s in samples]

        sites_vec = Comrade.sites(samples[1].instrument.lg)
        times_vec = Comrade.times(samples[1].instrument.lg)
        time_hrs = [t.t0 for t in times_vec]
        usites = unique(sites_vec)

        fig_g = Figure(size=(900, 500))
        ax_amp = Axis(fig_g[1, 1], xlabel="Time (hr)", ylabel="Amplitude gain", title="$(cfg.label): gains")
        ax_pha = Axis(fig_g[2, 1], xlabel="Time (hr)", ylabel="Phase gain (deg)")

        for (ic, site) in enumerate(usites)
            mask = sites_vec .== site
            ts = time_hrs[mask]

            # collect per-sample gain values at this site
            amps = hcat([exp.(lg[mask]) for lg in all_lg]...)   # ntimes × nsamples
            phas = hcat([rad2deg.(gp[mask]) for gp in all_gp]...)

            amp_med = vec(median(amps; dims=2))
            amp_lo  = vec(map(i -> quantile(amps[i,:], 0.16), axes(amps,1)))
            amp_hi  = vec(map(i -> quantile(amps[i,:], 0.84), axes(amps,1)))

            pha_med = vec(median(phas; dims=2))
            pha_lo  = vec(map(i -> quantile(phas[i,:], 0.16), axes(phas,1)))
            pha_hi  = vec(map(i -> quantile(phas[i,:], 0.84), axes(phas,1)))

            color = Makie.Cycled(ic)
            scatter!(ax_amp, ts, amp_med; color, label=string(site), markersize=6)
            rangebars!(ax_amp, ts, amp_lo, amp_hi; color)
            scatter!(ax_pha, ts, pha_med; color, label=string(site), markersize=6)
            rangebars!(ax_pha, ts, pha_lo, pha_hi; color)
        end

        hlines!(ax_amp, [1.0]; color=:gray, linestyle=:dash)
        hlines!(ax_pha, [0.0]; color=:gray, linestyle=:dash)
        axislegend(ax_amp; position=:lt)
        linkxaxes!(ax_amp, ax_pha)
        resize_to_layout!(fig_g)

        save("./figs/$(slug)__gains.png", fig_g)
        println("$(cfg.label) → gains saved to figs/$(slug)__gains.png")
    end
end
