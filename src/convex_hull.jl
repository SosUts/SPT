function convex_hull(
        df;
        idlabel = :TrackID,
        xlabel = :POSITION_X,
        ylabel = :POSITION_Y,
        framelabel = :FRAME
    )
    result = Matrix{Union{Float64,Nothing}}(nothing,
        Int(maximum(df[!, framelabel])), maximum(df[!, idlabel])
    )
    @inbounds for n in 1:maximum(df[!, idlabel])
        m = Matrix(df[df[!, idlabel] .== n, [xlabel, ylabel]])
        add_noise!(m)
        @simd for t in 3:size(m, 1)-2
            result[t-2, n] = chull(view(m, t-2:t+2, :)).area
        end
    end
    result
end