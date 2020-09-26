function gyration(
        df::DataFrame,
        idlabel::Symbol,
        xlabel::Symbol,
        ylabel::Symbol,
    )
    spt.add_dr!(df)
    N = maximum(df.TrackID)
    result = DataFrame(TrackID = Int[], value = Float64[])
    dr = mean(filter(!isnan, df.dr))
    @inbounds for n in 1:maximum(df[!, idlabel])
        data = extract(df, n, idlabel=idlabel, xlabel=xlabel, ylabel=ylabel)
        T = size(data, 1)
        x̄ = mean(data[:, 1])
        ȳ = mean(data[:, 2])
        c = 0.0
        @inbounds for t in 1:T
            x = abs2(data[t, 1] - x̄)
            y = abs2(data[t, 2] - ȳ)
            c += (x + y)
        end
        c *= ((sqrt(π/2)/dr)*1/T)
        push!(result, [n, c])
    end
    result
end