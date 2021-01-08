function convex_hull(df::AbstractArray)
    T, N = size(df, 1), size(df, 3)
    result = Matrix{Union{Float64,Nothing}}(
        nothing, T, N
    )
    @inbounds for n = 1:N
        @views m = convert.(Float64, remove_nothing(df[:, :, n]))
        # m = extract(df, n, id, [x, y])
        add_noise!(m)
        T = size(m, 1)
        # if T <= 90
        #     continue
        # end
        for t = 3:T-2
            result[t-2, n] = chull(m[t-2:t+2, :]).area
        end
    end
    result
end

function convex_hull(
    df::DataFrame;
    id = :TrackID,
    x = :POSITION_X,
    y = :POSITION_Y,
    frame = :FRAME2
)
    convex_hull(xy2matrix(df, id, x, y, frame))
end
