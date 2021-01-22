function add_dr!(
    df;
    idlabel = :TrackID,
    xlabel = :POSITION_X,
    ylabel = :POSITION_Y
)
    df[!, :dx] .= prepend!(diff(df[:, xlabel]), NaN)
    df[!, :dy] .= prepend!(diff(df[:, ylabel]), NaN)
    df.dr2 = abs2.(df.dx) + abs2.(df.dy)
    df.dr = sqrt.(df.dr2)

    @inbounds for i = 2:nrow(df)
        if df[i, idlabel] != df[i-1, idlabel]
            df[i, [:dx, :dy, :dr, :dr2]] .= NaN
        end
    end
end

function dr2matrix(df, idlabel, datalabel, framelabel)
    N = length(Set(df[!, idlabel]))
    dr = Matrix{Union{Nothing,Float64}}(nothing, maximum(df[!, framelabel]), N)
    @inbounds for (i, n) = enumerate(sort!(collect(Set(df[!, idlabel]))))
        m = filter(!isnan, df[df[!, idlabel] .== n, datalabel])
        @simd for t = 1:length(m)
            dr[t, i] = m[t]
        end
    end
    dr
end
