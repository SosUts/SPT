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

    @inbounds for i = 2:nrow(df)
        if df[i, idlabel] != df[i-1, idlabel]
            df[i, [:dx, :dy, :dr, :dr2]] .= NaN
        end
    end
end

# df[!, :TrackID] .= 1
# frame = :t
# for i = 2:nrow(df)
#     if (df[i, frame] != df[i-1, frame] + 1) && (df[i, :TrackID] == df[i-1, :TrackID])
#         df[i, :TrackID] = df[i-1, :TrackID] +1
#     end
#     if (df[i, frame] == df[i-1, frame] + 1) && (df[i, :TrackID] != df[i-1, :TrackID])
#         df[i, :TrackID] = df[i-1, :TrackID]
#     end
#     if (df[i, frame] != df[i-1, frame] + 1) && (df[i, :TrackID] != df[i-1, :TrackID])
#         df[i, :TrackID] = df[i-1, :TrackID] +1
#     end
# end