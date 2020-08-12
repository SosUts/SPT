function add_dr!(
        df::DataFrame;
        idlabel = :TrackID,
        xlabel = :POSITION_X,
        ylabel = :POSITION_Y
    )
    df[!, :dx] .= prepend!(diff(df[:, xlabel]), NaN)
    df[!, :dy] .= prepend!(diff(df[:, ylabel]), NaN)
    df.dr2 = abs2.(df.dx) + abs2.(df.dy)
    df.dr = sqrt.(df.dr2)

    @inbounds for i in 2:nrow(df)
        if df[i, idlabel] != df[i-1, idlabel]
            df[i, [:dx, :dy, :dr, :dr2]] .= NaN
        end
    end
end