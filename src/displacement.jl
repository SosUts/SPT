struct Diffusion{T1<:Real,T2<:Real,T3<:Real} <: ContinuousUnivariateDistribution
    D::T1
    δ::T2
    ϵ::T3
    Diffusion{T1,T2,T3}(D::T1, δ::T2, ϵ::T3) where {T1<:Real,T2<:Real,T3<:Real} =
        new{T1,T2,T3}(D, δ, ϵ)
end

function Diffusion(
    D::T1,
    δ::T2,
    ϵ::T3;
    check_args = true,
) where {T1<:Real,T2<:Real,T3<:Real}
    check_args && @check_args(Diffusion, D > zero(D) && δ > zero(δ) && ϵ >= zero(ϵ))
    return Diffusion{T1,T2,T3}(D, δ, ϵ)
end

@distr_support Diffusion 0 +Inf

# minimum(d::Diffusion) = 0.0
# maximum(d::Diffusion) = +Inf
cdf(d::Diffusion, x::Real) = 1 - exp(-x^2 / 4 * (d.D * d.δ + d.ϵ^2))
quantile(d::Diffusion, p) = sqrt(-4 * (d.D * d.δ + d.ϵ^2) * log(1 - p))
pdf(d::Diffusion, x::Float64) =
    0 ≤ x ? x / (2 * (d.D * d.δ + d.ϵ^2)) * exp(-x^2 / (4 * (d.D * d.δ + d.ϵ^2))) : zero(x)
logpdf(d::Diffusion, x::AbstractVector{<:Real}) =
    0 ≤ x ? log(x) - log(2 * (d.D * d.δ + d.ϵ^2)) - (x^2 / (4 * (d.D * d.δ + d.ϵ^2))) :
    zero(x)
# Distributions.pdf(d::Diffusion, x::AbstractArray{<:Real}) =
#     0 ≤ x ? x / 2(d.D * d.δ + d.ϵ^2) * exp(-x^2 / 4(d.D * d.δ + d.ϵ^2)) : zero(x)
quantile(d::Diffusion, x::AbstractVector{<:Real}) =
    sqrt(-4 * (d.D * d.δ + d.ϵ^2) * log(1 - x))
rand(d::Diffusion, rng::AbstractVector{<:Real}) = quantile(d::Diffusion, rng)


### MLE fitting
struct DiffusionStats <: SufficientStats
    r::Real # (weighted) sum of sqared r
    w::Float64 # total sample weight
    δ::Float64
    ϵ::Float64

    DiffusionStats(r::Real, w::Real, δ::Float64, ϵ::Float64) = new(r, w, δ, ϵ)
end

function Distributions.suffstats(
    ::Type{<:Diffusion},
    x::AbstractArray
)
    n = length(x)
    r = 0.0
    for i = 1:n
        @inbounds r += x[i]^2
    end
    DiffusionStats(r, Float64(n), 0.001, 0.0)
end

function Distributions.suffstats(d::Diffusion, x::AbstractArray)
    n = length(x)
    r = 0.0
    for i = 1:n
        @inbounds r += x[i]^2
    end
    DiffusionStats(r, Float64(n), d.δ, d.ϵ)
end

function Distributions.suffstats(::Type{<:Diffusion}, x::AbstractArray{T}, w::AbstractArray{Float64}) where T <: Real
    n = length(x)
    if length(w) != n
        throw(DimensionMismatch("Inconsistent argument dimensions."))
    end
    tw = w[1]
    r = 0.0
    for i = 1:n
        @inbounds wi = w[i]
        @inbounds xi = x[i]
        r += wi * xi^2
        tw += wi
    end
    DiffusionStats(r, tw, 0.001, 0.0)
end

function Distributions.fit_mle(::Type{<:Diffusion}, ss::DiffusionStats)
        # D = (ss.r - 4 * ss.w * ss.δ * ss.ϵ^2) / (4 * ss.w * ss.δ)
    D = ss.r / (4 * ss.w * ss.δ) - ss.ϵ^2 / ss.δ
    Diffusion(D, ss.δ, ss.ϵ)
end
