function _pre_calculation(df::DataFrame, average = :ensemble)
    @argcheck average in [:ensemble, :time_average]
    tmp_msd = DataFrame(
        TrackID = Int64[],
        msd = Float64[],
        corrected_msd = Float64[],
        delta_t = Int64[],
        n = Int64[],
    )
    @inbounds @fastmath for id in sort(collect(Set(df.TrackID)))
        column_list = [:POSITION_X, :POSITION_Y, :corrected_x, :corrected_y]
        tmp_df = convert(Matrix, df[df.TrackID.==id, column_list])
        track_length = size(tmp_df, 1)
        @inbounds @fastmath for delta_t = 1:track_length-1
            start = 1
            cumsum = 0.0
            corrected_cumsum = 0.0
            n = 0
            @inbounds @fastmath while start + delta_t <= track_length
                cumsum += norm(tmp_df[start+delta_t, 1:2] .- tmp_df[start, 1:2])^2
                corrected_cumsum += norm(tmp_df[start+delta_t, 3:4] .- tmp_df[start, 3:4])^2
                start += 1
                n += 1
                if average == :ensemble
                    break
                end
            end
            cumsum /= n
            corrected_cumsum /= n
            push!(tmp_msd, [id, cumsum, corrected_cumsum, delta_t, n])
        end
    end
    tmp_msd
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
        tmp_msd = df[df.delta_t.==i, :msd]
        tmp_corrected_msd = df[df.delta_t.==i, :corrected_msd]
        push!(
            msd,
            [
                i,
                mean(tmp_msd),
                mean(tmp_corrected_msd),
                total_n,
                std(tmp_msd),
                sem(tmp_msd),
            ],
        )
    end
    msd
end

"""

"""
function mean_square_disaplcement(df::DataFrame, ; average = :ensemble)
    @argcheck average in [:ensemble, :time_average]
    tmp = _pre_calculation(df, average)
    return _average_msd(tmp), tmp
end

function fit_msd(df, ; max_time::Int64 = 10, loc_error::Float64 = 0.03, p0 = [1.0, 1.0])
    @. model(x, p) = 4 * p[1] * x^p[2] + 4 * 0.03^2
    fit = curve_fit(model, df.delta_t[1:max_time], df.msd[1:max_time], p0)
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
    xlabel("Δt (sec)", fontsize = 14, family = "Arial")
    xticks([0.05, 0.1, 0.15, 0.2, 0.25], fontsize = 12)

    ylim(0,)
    ylabel("MSD (μm^2 / sec)", fontsize = 14, family = "Arial")
    yticks(fontsize = 12)

    legend(bbox_to_anchor = (1.02, 1.0), loc = 2, borderaxespad = 0.0, fontsize = 14)
    if save_fig
        savefig("tamsd_dia0.25.png", bbox_inches = "tight", dpi = 800)
    end
end
