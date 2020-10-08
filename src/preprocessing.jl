function add_dr!(df::DataFrame; id = :TrackID, xlabel = :POSITION_X, ylabel = :POSITION_Y)
    df[!, :dx] .= prepend!(diff(df[:, xlabel]), NaN)
    df[!, :dy] .= prepend!(diff(df[:, ylabel]), NaN)
    df.dr2 = abs2.(df.dx) + abs2.(df.dy)
    df.dr = sqrt.(df.dr2)

    @inbounds for i = 2:nrow(df)
        if df[i, id] != df[i-1, id]
            df[i, [:dx, :dy, :dr, :dr2]] .= NaN
        end
    end
end
