function ensemble_average_msd(
        data::AbstractMatrix{Float64},
    )
    T = size(data, 1)
    ? = 1
    c = 0.0
    @inbounds for t in 1:(T-1)
        c += distance(data, t, 1)
    end
    c /= (T-1)
end

function _msd_pre_calculation(
    df::DataFrame,
    average::Symbol = :ensemble_average
    )
    @argcheck average in [:ensemble_average, :time_average]
    result = DataFrame(
        TrackID = Int64[],
        n = Int64[],
        ? = Int64[],
        msd = Float64[]
    )
    @inbounds for id in sort(collect(Set(df.TrackID)))
    column_list = [:POSITION_X, :POSITION_Y, :corrected_x, :corrected_y]
    data = convert(Matrix, df[df.TrackID .== id, column_list])
    T = size(tmp_df, 1)
    for ? in 1:T-1
        c = 0.0
        n = 1
        for t in 1:T-?
            c += distance(data, t, ?)
            n += 1
            if average == :ensemble_average
                break
            end
        end
        c /= n
        push!(result, [id, n, ?, c])
    end
    result
end

function _average_msd(df::DataFrame)
    msd = DataFrame(
        delta_t = Float64[],
        msd = Float64[],
        corrected_msd = Float64[],
        n = Int64[],
        std = Float64[],
        sem = Float64[],
    )
    @inbounds @simd for i = 1:maximum(df.delta_t)
        total_n = sum(df[df.delta_t.==i, :n])
        result = df[df.delta_t.==i, :msd]
        tmp_corrected_msd = df[df.delta_t.==i, :corrected_msd]
        push!(
            msd,
            [
                i,
                mean(result),
                mean(tmp_corrected_msd),
                total_n,
                std(result),
                sem(result),
            ],
        )
    end
    msd
end

"""
    mean_square_disaplcement(df; average=[:ensemble_average, time_average], non_averaged::Bool) -> DataFrame, [DataFrame]

Compute mean_square_disaplcement over files.

**Example**
```julia
using SPT

```
"""
function mean_square_disaplcement(
    df::DataFrame;
    average = :ensemble_average,
    non_averaged::Bool = true
    )
    @argcheck average in [:ensemble_average, :time_average]
    tmp = _msd_pre_calculation(df, average)
    if non_averaged
        return _average_msd(tmp), tmp
    end
    _average_msd(tmp)
end

function fit_msd(
    df;
    max_time::Int64 = 10,
    loc_error::Float64 = 0.03,
    p0::AbstractVector = [1.0, 1.0]
    )
    @. model(x, p) = 4 * p[1] * x^p[2] + 4 * loc_error^2
    fit = curve_fit(model, df.delta_t[1:max_time].*0.022, df.msd[1:max_time], p0)
    fit.param
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
    # if save_fig
    #     savefig("tamsd_dia0.5.png", bbox_inches = "tight", dpi = 800)
    # end
end
