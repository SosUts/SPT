function asymmetry(df, id, x, y)
    result = DataFrame(TrackID = Int[], len = Int[], asymmetry = Float64[])
    @inbounds for n = minimum(df[!, id]):maximum(df[!, id])
        m = extract(df, n, id, [x, y])
        T = Matrix{Float64}(undef, 2, 2)
        mean_x = @views mean(m[:, 1])
        mean_y = @views mean(m[:, 2])
        t1 = 0.0
        t2 = 0.0
        t3 = 0.0
        N = size(m, 1)
        for t = 1:N
            _x = m[t, 1] - mean_x
            _y = m[t, 2] - mean_y
            t1 += abs2(_x)
            t2 += _x * _y
            t3 += abs2(_y)
        end
        t1 /= N
        t2 /= N
        t3 /= N
        T[1, 1] = t1
        T[1, 2] = t2
        T[2, 1] = t2
        T[2, 2] = t3
        λ1, λ2 = eigvals(T)
        A = -log(1 - abs2(λ1 - λ2)/(2*abs2(λ1 + λ2)))
        push!(result, [Int(n), Int(N), A])
    end
    result
end

function non_gaussian_parameter(df, x, frame=:FRAME2)
    Δ = diff(df[!, x])
    prepend!(Δ, NaN)
    Δ[findall(x -> x == 1, df[!, frame])] .= NaN
    filter!(!isnan, Δ)
    a = mean(Δ.^4)
    b = mean(abs2.(Δ))^2
    a/3b - 1
end

function efficiency(df, id, x, y)
    result = DataFrame(TrackID = Int[], value = Float64[])
    @inbounds for n in 1:maximum(df[!, id])
        m = extract(df, n, id, [x, y])
        c = 0.0
        T = size(m, 1)
        for t in 1:T-1
            c += squared_displacement(m, t, 1)
        end
        c /= (T-1)
        c₀ = squared_displacement(m, 1, T-1)
        push!(result, [n, c₀/c])
    end
    result
end

function _largest_distance(m)
    T = size(m, 1)
    d = displacement(m, 1)
    for δ = 1:T-1
        for t = 1:T-δ
            dₜ = displacement(m, t, δ)
            if dₜ > d
                d = dₜ
            end
        end
    end
    d
end

function _total_length(m)
    T = size(m, 1)
    c = 0.0
    for t = 1:T-1
        c += displacement(m, t, 1)
    end
    c
end

function fractal_dimension(df, id, x, y)
    result = DataFrame(TrackID = Int[], value = Float64[])
    @inbounds @simd for n in minimum(df[!, id]):maximum(df[!, id])
        m = extract(df, n, id, [x, y])
        N = size(m, 1)-1
        d = _largest_distance(m)
        L = _total_length(m)
        push!(result, [n, log(N)/log(N*d/L)])
    end
    result
end

function gaussianity(
    df::DataFrame,
    id::Symbol = :TrackID,
    x::Symbol = :POSITION_X,
    y::Symbol = :POSITION_Y,
)
    gaussianity = DataFrame(TrackID = Int64[], gaussianity = Float64[], delta_t = Int64[], n = Int64[])
    @inbounds for n in sort(collect(Set(df[!, id])))
        m = extract(df, Int(n), id, [x, y])
        T = size(m, 1)
        for δ = 1:T-1
            r⁴ = 0.0
            r² = 0.0
            @simd for t = 1:(T-δ)
                r⁴ += abs2(squared_displacement(m, t, δ))
                r² += squared_displacement(m, t, δ)
            end
            r⁴ /= (T - δ)
            r² /= (T - δ)
            push!(gaussianity, [n, r⁴/(2*abs2(r²)) - 1, δ, T - δ])
        end
    end
    gaussianity
end