function _pre_calculation(df::DataFrame)
    #=
    BenchmarkTools.Trial: 
  memory estimate:  525.38 MiB
  allocs estimate:  11288180
  --------------
  minimum time:     1.728 s (1.97% GC)
  median time:      1.801 s (2.91% GC)
  mean time:        1.843 s (2.57% GC)
  maximum time:     1.999 s (2.62% GC)
  --------------
  samples:          3
  evals/sample:     1
    =#
    tmp_msd = DataFrame(
        TrackID = Int64[], MSD = Float64[], Δt = Int64[], n=Int64[]
    )
    @inbounds @simd for id in sort(collect(Set(df.TrackID)))
        tmp_df = convert(Matrix, df[df.TrackID.==id, [:POSITION_X, :POSITION_Y]])
        track_length = size(tmp_df, 1)
        for Δt in 1:track_length-1
            start = 1
            cumsum = 0.0
            n = 0
            while start+Δt <= track_length
                cumsum += norm(tmp_df[start+Δt,:]-tmp_df[start,:])^2
                start += 1
                n += 1
            end
            cumsum /= n
            push!(
                tmp_msd, [
                    id, cumsum, Δt, n
                ]
            )
        end
    end
    tmp_msd
end

function _average_msd(df::DataFrame)
    msd = DataFrame(
            Δt=Float64[], MSD=Float64[], n=Int64[], std=Float64[], sem=Float64[]
        )
    @inbounds @simd for i in 1:maximum(tmp_msd.Δt)
        total_n = sum(df[df.Δt.==i, :n])
        tmp = df[df.Δt.==i, :]
        push!(
            msd,[
                i*1/45, mean(tmp.MSD), total_n, std(tmp.MSD), sem(tmp.MSD)
            ]
        )
    end
    msd
end

function mean_square_disaplcement(df::DataFrame)
    tmp = _pre_calculation(df)
     _average_msd(tmp)
end

function fit_msd(df, ;max_time::Int64=10)
    @. model(x, p) = 4*p[1]*x^p[2] + 4*0.03^2
    fit = curve_fit(model, df.Δt[1:max_time], df.MSD[1:max_time], [1.0, 1.0])
    D, α = fit.param
    return D, α
end