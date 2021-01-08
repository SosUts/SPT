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
function _dynamical_functional(data::AbstractVector, n::Int, ω)
    @argcheck (n >= 1) && (ω >= 0)
    T = length(data)

    e1 = 0.0
    @inbounds for k = 1:(T-n) # kは開始地点
        @views e1 += exp(ω * im * (data[k+n] - data[k]))
    end
    e1 /= (T - n)

    e2 = 0.0
    @inbounds for k = 1:T
        @views e2 += exp(ω * im * (data[k] - data[1]))
    end
    e2 = abs2(e2) / (T * (T - 1))
    e1 - e2 + (1 / (T + 1))
end

function dynamical_functional(
    df::DataFrame,
    id::Symbol = :TrackID,
    x::Symbol = :POSITION_X;
    # frame::Symbol = :FRAME2;
    ω = 1,
)
    N = maximum(df[!, id])
    # T = Int(maximum(df[!, frame])) - 1
    T = 1000-1
    E = Matrix{Union{ComplexF64,Nothing}}(nothing, T, N)
    @inbounds for n = 1:N
        data = spt.extract(df, n, id, x)
        _T = size(data, 1)-1
        @simd for t = 1:_T
            E[t, n] = spt._dynamical_functional(data, t, ω)
        end
    end
    E
end

function ergodicity_estimator(E)
    T, N = size(E)
    F = Matrix{Union{ComplexF64,Nothing}}(nothing, T, N)
    @inbounds for n = 1:N
        @views @simd for t = 1:size(filter(!isnothing, E[:, n]), 1)
            @views F[t, n] = sum(E[1:t, n])/t
        end
    end
    F
end
