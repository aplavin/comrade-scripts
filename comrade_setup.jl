import DataManipulation: getproperties, StructArray

# Workarounds for Comrade/TransformVariables/VLBIImagePriors version incompatibilities:
# 1) inverse_eltype ambiguity between AbstractInstrumentTransform and AbstractTransform
let TV = Comrade.HypercubeTransform.TransformVariables
    TV.inverse_eltype(t::Comrade.AbstractInstrumentTransform, ::Type{T}) where {T} = eltype(T)
end
# 2) robust_eltype removed from TransformVariables but still used by VLBIImagePriors
@eval Comrade.HypercubeTransform.TransformVariables robust_eltype(x) = eltype(x)

struct ByFrequency end

function _aggfreq(freq_specs)
    mean_freq = sum(fs.freq for fs in freq_specs) / length(freq_specs)
    total_width = sum(fs.width for fs in freq_specs)
    return VLBIFiles.FrequencyWindow(0, 0, mean_freq, total_width, 1, 1)
end

function VLBIData.average_data(::ByFrequency, uvtbl; avgvals=VLBIData.Uncertain.weightedmean)
    uvtbl = StructArray(uvtbl)
    merged_fs = _aggfreq(unique(uvtbl.freq_spec))
    const_part = @p getproperties(uvtbl) (@delete __[(:source, :freq_spec, :stokes, :scan_id, :value, :spec, :count, :datetime)]) filter(allequal) map(uniqueonly)
    NT = VLBIData.intersect_nt_type(eltype(uvtbl), NamedTuple{(:source, :stokes, :scan_id)})
    @p begin
        uvtbl
        groupview_vg((; bl=Baseline(_), NT(_)...))
        map((;
            const_part...,
            delete(key(_), @o _.bl)...,
            freq_spec=merged_fs,
            count=length(_),
            value=avgvals(_.value),
            datetime=_.datetime[1],
            spec=VLBIData.aggspec(key(_).bl, _.spec),
        ))
        StructArray()
    end
end

function load_uvdata(datafile)
    obs = VLBIFiles.load(datafile)
    uvtbl_orig = VLBIFiles.uvtable(obs)
    uvtbl_I_avg = @p let
        uvtbl_orig
        VLBI.rescale_visibility_errors(VLBI.CoherentAverageScatter())
        VLBI.average_data(VLBI.GapBasedScans())
        VLBI.uvtable_values_to(VLBI.IPol)
        VLBI.average_data(ByFrequency())
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