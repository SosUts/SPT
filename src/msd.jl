function pre_calculation(df::DataFrame, ; time_average::Bool = false)
    tmp_msd = DataFrame(
        TrackID = Int64[], msd = Float64[], corrected_msd = Float64[],
        delta_t = Int64[], n = Int64[]
    )
    @inbounds @fastmath for id in sort(collect(Set(df.TrackID)))
        column_list = [
            :POSITION_X, :POSITION_Y, :corrected_x, :corrected_y
        ]
        tmp_df = convert(Matrix, df[df.TrackID .== id, column_list])
        track_length = size(tmp_df, 1)
        @inbounds @fastmath for delta_t in 1:track_length - 1
            start = 1
            cumsum = 0.0
            corrected_cumsum = 0.0
            n = 0
            @inbounds @fastmath while start + delta_t <= track_length
                cumsum +=
                    norm(tmp_df[start + delta_t, 1:2] .- tmp_df[start, 1:2])^2
                corrected_cumsum +=
                    norm(tmp_df[start + delta_t, 3:4] .- tmp_df[start, 3:4])^2
                start += 1
                n += 1
                if time_average
                    break
                end
            end
            cumsum /= n
            corrected_cumsum /= n
            push!(
                tmp_msd, [
                    id, cumsum, corrected_cumsum, delta_t, n
                ]
            )
        end
    end
    tmp_msd
end

function average_msd(df::DataFrame)
    msd = DataFrame(
        delta_t = Float64[],
        msd = Float64[],
        n = Int64[],
        std = Float64[],
        sem = Float64[],
    )
    @inbounds @simd for i = 1:maximum(df.delta_t)
        total_n = sum(df[df.delta_t.==i, :n])
        tmp_msd = df[df.delta_t.==i, :msd]
        push!(msd, [i * 1 / 45, mean(tmp_msd), total_n, std(tmp_msd), sem(tmp_msd)])
    end
    msd
end

function mean_square_disaplcement(df::DataFrame, ; time_average = false)
    if time_average
        tmp = pre_calculation(df)
    else
        tmp = pre_calculation(df, time_average = true)
    end
    return average_msd(tmp), tmp
end

function fit_msd(df, ; max_time::Int64 = 10, loc_error::Float64 = 0.03, p0 = [1.0, 1.0])
    @. model(x, p) = 4 * p[1] * x^p[2] + 4 * 0.03^2
    fit = curve_fit(model, df.delta_t[1:max_time], df.msd[1:max_time], p0)
    fit.param
end
