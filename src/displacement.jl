mutable struct Diffusion{T <: Real} <: ContinuousUnivariateDistribution
    D::T
    δ::T
    ϵ::T
    Diffusion{T}(D::T, δ::T, ϵ::T) where {T <: Real} = new{T}(D, δ, ϵ)
end

function Diffusion(D::T, δ::T, ϵ; check_args = true) where {T <: Real}
    check_args && Distributions.@check_args(
        Diffusion, D > zero(D) && δ > zero(δ) && ϵ >= zero(ϵ)
    )
    return Diffusion{T}(D, δ, ϵ)
end

Distributions.@distr_support Diffusion 0 +Inf

Distributions.minimum(d::Diffusion) = 0
Distributions.maximum(d::Diffusion) = +Inf
Distributions.cdf(d::Diffusion, x::Real) where T <: Real = 
    1 - exp(-x^2 / 4(d.D * d.δ + d.ϵ^2))
Distributions.quantile(d::Diffusion, p) = sqrt(-4(d.D * d.δ + d.ϵ^2) * log(1 - p))
Distributions.pdf(d::Diffusion, x::Float64) where T <: Real =
    0 ≤ x ? x / 2(d.D * d.δ + d.ϵ^2) * exp(-x^2 / 4(d.D * d.δ + d.ϵ^2)) : zero(T)
Distributions.logpdf(d::Diffusion, x::AbstractVector{<:Real}) where T <: Real = 
    0 ≤ x ? log(x) - log(2(d.D * d.δ + d.ϵ^2)) - x^2 / (4(d.D * d.δ + d.ϵ^2)) : zero(T)
Distributions.quantile(d::Diffusion, x::AbstractVector{<:Real}) = sqrt(-4(d.D * d.δ + d.ϵ^2) * log(1 - x))
Distributions.rand(d::Diffusion, rng::AbstractVector{<:Real}) = 
    Distributions.quantile(d::Diffusion, rng)


### MLE fitting
mutable struct DiffusionStats <: SufficientStats
    r::Real # (weighted) sum of sqared r
    w::Float64 # total sample weight
end

function Distributions.suffstats(::Type{<:Diffusion}, x::AbstractArray{T}) where T <: Real
    n = length(x)
    _δ = d.δ
    _ϵ = d.ϵ
    r = 0
    for i = 1:n
        @inbounds r += x[i]^2
    end
    DiffusionStats(r, n)
end

function Distributions.suffstats(::Type{<:Diffusion}, x::AbstractArray{T}, w::AbstractArray{Float64}) where T <: Real
    n = length(x)
    if length(w) != n
        throw(DimensionMismatch("Inconsistent argument dimensions."))
    end
    _δ = 0.001
    _ϵ = 0.03
    tw = w[1]
    r = 0
    for i = 1:n
        @inbounds wi = w[i]
        @inbounds xi = x[i]
        r += wi * xi^2
        tw += wi
    end
    DiffusionStats(r, tw)
end

function Distributions.fit_mle(d::Diffusion, ss::DiffusionStats)
    _δ = d.δ
    _ϵ = d.ϵ
    a = (ss.r * _δ - 4 * ss.w * _δ * _ϵ^2) / (4 * ss.w * _δ^2)
    Diffusion(a, _δ, _ϵ)
end