function convex_hull(df; id = :TrackID, x = :POSITION_X, y = :POSITION_Y, frame = :FRAME)
    result = Matrix{Union{Float64,Nothing}}(
        nothing,
        Int(maximum(df[!, frame])),
        maximum(df[!, id]),
    )
    @inbounds for n = 1:maximum(df[!, id])
        m = data = extract(df, n, id, x, y)
        add_noise!(m)
        @simd for t = 3:size(m, 1)-2
            result[t-2, n] = chull(view(m, t-2:t+2, :)).area
        end
    end
    result
end
