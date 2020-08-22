
"""
    mean_square_disaplcement(df; method=[:ensemble_average, time_average], non_averaged::Bool) -> DataFrame, [DataFrame]

Compute mean_square_disaplcement over files.

**Example**
```julia
using SPT

```
"""
function ensemble_time_average_msd(
    df::DataFrame,
    xlabel::Symbol,
    ylabel::Symbol;
    return_tamsd::Bool = false
    )
    ta_msd = time_average_msd(df, xlabel, ylabel)
    return_tamsd ? (ensemble_tamsd(ta_msd), ta_msd) : ensemble_tamsd(ta_msd)
end

function time_average_msd(
    df::DataFrame,
    xlabel::Symbol,
    ylabel::Symbol
    )
    tamsd = DataFrame(
        TrackID = Int64[],
        msd = Float64[],
        delta_t = Int64[],
        n = Int64[],
    )
    @inbounds for id in sort(collect(Set(df.TrackID)))
        data = convert(Matrix, df[df.TrackID.==id, [xlabel, ylabel]])
        T = size(data, 1)
        @inbounds for δ in 1:T-1
            c = 0.0
            @inbounds for t in 1:(T-δ)
                c += squared_displacement(data, t, δ)
            end
            c /= (T-δ)
            push!(tamsd, [id, c, δ, T-δ])
        end
    end
    tamsd
end

function ensemble_msd(
    df::DataFrame,
    xlabel::Symbol,
    ylabel::Symbol
    )
    eamsd = DataFrame(
        TrackID = Int64[],
        msd = Float64[],
        n = Int[],
        delta_t = Int64[]
    )
    @inbounds for id in sort(collect(Set(df.TrackID)))
        data = convert(Matrix, df[df.TrackID.==id, [xlabel, ylabel]])
        T = size(data, 1)
        @inbounds for δ in 1:T-1
            push!(eamsd,
            [id, squared_displacement(data, 1, δ), 1, δ]
            )
        end
    end
    eamsd
end

function ensemble_tamsd(df::DataFrame)
    eatamsd = DataFrame(
        delta_t = Float64[],
        msd = Float64[],
        n = Int64[],
        std = Float64[],
        sem = Float64[],
    )
    @inbounds @simd for i = 1:maximum(df.delta_t)
        data = df[df.delta_t .== i, :]
        push!(
            eatamsd, [
                i, mean(data.msd), sum(data.n), std(data.msd),sem(data.msd),
            ],
        )
    end
    eatamsd
end

function fit_msd(
    df;
    max_time::Int64 = 10,
    loc_error::Float64 = 0.03,
    p0::AbstractVector = [1.0, 1.0]
    )
    @argcheck loc_error >= 0.0
    @. model(x, p) = 4 * p[1] * x^p[2] + 4 * loc_error^2
    fit = curve_fit(
        model, df.delta_t[1:max_time],
        df.msd[1:max_time], p0, lower=[0.0, 0.0], upper=[10.0, 2]
        )
    fit.param
end

function fit_msd(df::DataFrame, δ::Int)
    @argcheck δ >= 1
    # @argcheck method in [:arithmetic, :geometric]
    log(mean(df[df.delta_t .== δ, :msd])/mean(df[df.delta_t .== 1, :msd]))/log(δ/1)
end

function plot_msd(grouped_df; maxt = 10, save_fig = false)
    for i = 1:length(grouped_df)
        result = []
        tmp_df = grouped_df[i]
        for t = 1:maxt
            a = tmp_df[tmp_df.delta_t.==t, :]
            push!(result, [t, mean(a.msd), sum(a.n), sem(a.msd)])
        end
        insert!(result, 1, [0, 0, 0, 0])
        result = reduce(vcat, result')
        genotype = tmp_df[1, :Genotype]
        plot(result[:, 1] * (1 / 45), result[:, 2], label = "$genotype")
        fill_between(
            result[:, 1] * (1 / 45),
            result[:, 2] .- result[:, 4],
            result[:, 2] .+ result[:, 4],
            alpha = 0.2,
        )
    end

    xlim(0,)
    xlabel("delta t (sec)", fontsize = 14, family = "Arial")
    xticks([0.05, 0.1, 0.15, 0.2, 0.25], fontsize = 12)

    ylim(0,)
    ylabel("MSD (μm^2 / sec)", fontsize = 14, family = "Arial")
    yticks(fontsize = 12)

    legend(bbox_to_anchor = (1.02, 1.0), loc = 2, borderaxespad = 0.0, fontsize = 14)
    if save_fig
        savefig("tamsd_dia0.5.png", bbox_inches = "tight", dpi = 800)
    end
end

# function ensemble_msd(
#         data::AbstractMatrix{Float64},
#     )
#     T = size(data, 1)
#     δ = 1
#     c = 0.0
#     @inbounds for t in 1:(T-1)
#         c += distance(data, 1)
#     end
#     c /= (T-1)
# end