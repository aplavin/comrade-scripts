import DataManipulation: getproperties, StructArray

function load_uvdata(datafile)
    obs = VLBIFiles.load(datafile)
    uvtbl_orig = VLBIFiles.uvtable(obs)
    uvtbl_I_avg = @p let
        uvtbl_orig
        VLBI.rescale_visibility_errors(VLBI.CoherentAverageScatter())
        VLBI.average_data(VLBI.GapBasedScans())
        VLBI.uvtable_values_to(VLBI.IPol)
        VLBI.average_data(VLBI.ByFrequency())
    end
    cmrd_data = Comrade.extract_table(uvtbl_I_avg; antennas=only(obs.ant_arrays).antennas |> collect)
    return (; obs, uvtbl_orig, uvtbl_I_avg, cmrd_data)
end

function sky(θ, metadata)
    (; c, σimg, ftot) = θ
    (; mimg) = metadata
    rast = ftot .* apply_fluctuations(CenteredLR(), mimg, σimg .* c.params)
    img = ContinuousImage(rast, BSplinePulse{3}())
    x0, y0 = centroid(rast)
    return shifted(img, -x0, -y0)
end

function build_instrument(; σamp, σphase)
    @assert isnothing(σamp) == isnothing(σphase)
    if isnothing(σamp)
        return nothing
    end
    G = SingleStokesGain(x -> @fastmath exp(x.lg + im * x.gp))
    κphase = inv(σphase^2)
    intprior = (
        lg = ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, σamp))),
        gp = ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, κphase)),
                        refant=SEFDReference(0.0), phase=true;
                        space=IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, κphase)))
    )
    return InstrumentModel(G, intprior)
end
