"""
    AbstractHMM{F<:VariateForm}

A custom HMM type must at-least implement the following interface:
```julia
struct CustomHMM{F,T} <: AbstractHMM{F}
    a::AbstractVector{T}               # Initial state distribution
    A::AbstractMatrix{T}               # Transition matrix
    B::AbstractVector{Distribution{F}} # Observations distributions
    # Optional, custom, fields ....
end
```
"""
abstract type AbstractHMM{F<:VariateForm} end

"""
    HMM([a, ]A, B) -> HMM

Build an HMM with transition matrix `A` and observation distributions `B`.  
If the initial state distribution `a` is not specified, a uniform distribution is assumed. 

Observations distributions can be of different types (for example `Normal` and `Exponential`),  
but they must be of the same dimension.

**Arguments**
- `a::AbstractVector{T}`: initial probabilities vector.
- `A::AbstractMatrix{T}`: transition matrix.
- `B::AbstractVector{<:Distribution{F}}`: observations distributions.

**Example**
```julia
using Distributions, HMMBase
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
```
"""
struct HMM{F,T} <: AbstractHMM{F}
    a::Vector{T}
    A::Matrix{T}
    B::Vector{Distribution{F}}
    HMM{F,T}(a, A, B) where {F,T} = assert_hmm(a, A, B) && new(a, A, B)
end

HMM(
    a::AbstractVector{T},
    A::AbstractMatrix{T},
    B::AbstractVector{<:Distribution{F}},
) where {F,T} = HMM{F,T}(a, A, B)
HMM(A::AbstractMatrix{T}, B::AbstractVector{<:Distribution{F}}) where {F,T} =
    HMM{F,T}(ones(size(A)[1]) / size(A)[1], A, B)

"""
    assert_hmm(a, A, B)

Throw an `ArgumentError` if the initial state distribution and the transition matrix rows does not sum to 1,
and if the observation distributions do not have the same dimensions.
"""
function assert_hmm(a::AbstractVector, A::AbstractMatrix, B::AbstractVector{<:Distribution})
    @argcheck isprobvec(a)
    @argcheck istransmat(A)
    @argcheck all(length.(B) .== length(B[1])) ArgumentError("All distributions must have the same dimensions")
    @argcheck length(a) == size(A, 1) == length(B)
    return true
end

"""
    issquare(A) -> Bool

Return true if `A` is a square matrix.
"""
issquare(A::AbstractMatrix) = size(A, 1) == size(A, 2)

"""
    istransmat(A) -> Bool

Return true if `A` is square and its rows sums to 1.
"""
istransmat(A::AbstractMatrix) =
    issquare(A) && all([isprobvec(A[i, :]) for i = 1:size(A, 1)])

isequal(h1::AbstractHMM, h2::AbstractHMM) =
    (h1.a == h2.a) && (h1.A == h2.A) && (h1.B == h2.B)

function rand(
    rng::AbstractRNG,
    hmm::AbstractHMM{Univariate},
    z::AbstractArray{<:Integer},
    T::Integer,
    N::Integer;,
)
    y = Array{Float64}(undef, size(z, 1), size(z, 2))
    for n = 1:N
        for t = 1:T
            y[t, n] = rand(rng, hmm.B[z[t]])
        end
    end
    y
end

function rand(rng::AbstractRNG, hmm::AbstractHMM, T::Integer, N::Integer; seq = false)
    z = Matrix{Int}(undef, T, N)
    for n = 1:N
        z[1, n] = rand(rng, Categorical(hmm.a))
        for t = 2:T
            z[t, n] = rand(rng, Categorical(hmm.A[z[t-1, n], :]))
        end
    end
    y = rand(rng, hmm, z, T, N)
    seq ? (z, y) : y
end

Base.rand(hmm::AbstractHMM, T::Integer, N::Integer; kwargs...) =
    rand(GLOBAL_RNG, hmm, T, N; kwargs...)

Base.rand(hmm::AbstractHMM, z::AbstractArray{<:Integer}) =
    rand(GLOBAL_RNG, hmm, size(z, 1), size(z, 2))
