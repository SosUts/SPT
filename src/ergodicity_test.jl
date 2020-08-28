function velocity_normalization_constant(
        data::AbstractVector
    )
    T = length(data)
    c = 0.0
    for δ in 1:T-1
        c += velocity(data, δ)^2
    end
    c /= (T-1)
    sqrt(c)
end

function velocity(
        data::AbstractVector,
        n::Int
    )
    data[n+1] - data[n]
end

# n（タイムラグ）
function dynamical_functional(
        data::AbstractVector,
        n::Int,
        ω::Int
    )
    # @argcheck (n >= 1) && (ω >= 0)
    T = length(data)

    e1 = 0.0
    for k in 1:(T-n) # kは開始地点
        e1 += exp(ω*im*(data[k+n]-data[k]))
    end
    e1 /= (T-n)

    e2 = 0.0
    for k in 1:T
        e2 += exp(ω*im*(data[k]-data[1]))
    end
    e2 = abs2(e2)/(T*(T-1))
    # e2 /= (T^2+T)
    e1-e2+(1/(T+1))
end
