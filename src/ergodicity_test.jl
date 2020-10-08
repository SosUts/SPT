function velocity_normalization_constant(data::AbstractVector)
    T = length(data)
    c = 0.0
    for δ = 1:T-1
        c += velocity(data, δ)^2
    end
    c /= (T - 1)
    sqrt(c)
end

function velocity(data::AbstractVector, n::Int)
    data[n+1] - data[n]
end

# n（タイムラグ）
function _dynamical_functional(data::AbstractVector, n::Int, ω::Int)
    @argcheck (n >= 1) && (ω >= 0)
    T = length(data)

    e1 = 0.0
    @inbounds for k = 1:(T-n) # kは開始地点
        e1 += exp(ω * im * (data[k+n] - data[k]))
    end
    e1 /= (T - n)

    e2 = 0.0
    @inbounds for k = 1:T
        e2 += exp(ω * im * (data[k] - data[1]))
    end
    e2 = abs2(e2) / (T * (T - 1))
    e1 - e2 + (1 / (T + 1))
end

function dynamical_functional(
    df::DataFrame,
    id::Symbol = :TrackID,
    x::Symbol = :POSITION_X,
    frame::Symbol = :FRAME2;
    ω::Int = 1,
)
    N = maximum(df[!, id])
    T = Int(maximum(df[!, frame])) - 1
    # T = 1000-1
    e_real = Matrix{Union{Float64,Nothing}}(nothing, T, N)
    e_im = Matrix{Union{Float64,Nothing}}(nothing, T, N)
    f_real = Matrix{Union{Float64,Nothing}}(nothing, T, N)
    f_im = Matrix{Union{Float64,Nothing}}(nothing, T, N)
    @inbounds for n = 1:N
        data = spt.extract(df, n, id, x)
        _T = size(data, 1) - 1
        if _T > 45
            continue
        end
        e = Vector{Complex}(undef, _T)
        f = Vector{Complex}(undef, _T)
        @simd for t = 1:_T
            e[t] = spt._dynamical_functional(data, t, ω)
        end
        @simd for t = 1:_T
            f[t] = mean(e[1:t])
        end
        @simd for t = 1:_T
            e_real[t, n] = reim(e[t])[1]
            e_im[t, n] = reim(e[t])[2]
            f_real[t, n] = reim(f[t])[1]
            f_im[t, n] = reim(f[t])[2]
        end
    end
    return e_real, e_im, f_real, f_im
end

function mean_df(x)
    T = size(x, 1)
    result = Vector{Float64}(undef, T)
    @inbounds @simd for t = 1:T
        result[t] = mean(filter(!isnothing, x[t, :]))
    end
    result
end
