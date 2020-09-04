#= original implementation from
Single-Particle Diffusion Characterization by Deep Learning
Biophys J. 2019 Jul 23;117(2):185-192. doi: 10.1016/j.bpj.2019.06.015. Epub 2019 Jun 22.
Naor Granik et. al.
https://github.com/AnomDiffDB/DB
=#

function mittag_leffer_rand(β::Real, N::Int, γ::Real)
    t = -log.(rand(N))
    u = rand(N)
    w = sin(β*π)./tan.(β*π.*u) .- cos(β*π)
    t .*= w./β
    t .*= γ
end

function symmetric_alpha_levy(α::Real, N::Int, γ::Real)
    u = rand(N)
    v = rand(N)
    ϕ = π .* (v.-0.5)
    w = sin.(α.*ϕ) ./ cos.(ϕ)
    z = -log.(u) .* cos.(ϕ)
    z ./= cos.((1-α).*ϕ)
    γ.*w.*z.^(1-(1/α))
end

find_nearest(array::AbstractArray, value) = argmin(abs.(array .- value))

"""
    ctrw(α, γ, T, N) -> Vector{Float}, Vector{Float}, Vector{Float64}

Return a simulated CTRW trajectory presented by Fulger et. al., Phys. Rev. E(2008).

**Arguments**

- `α`: exponent of the waiting time distribution function.
- `β`: scale parameter for the mittag-leffler and alpha stable distributions.
- `N`: number of points to generate.
- `T`: end time.
"""
function ctrw(α::Real, γ::Real, T::Int, N::Int)
    jumpsX = mittag_leffer_rand(α, N, γ)
    rawTimeX = cumsum(jumpsX)
    tX = rawTimeX*T / maximum(rawTimeX)
    jumpsY = mittag_leffer_rand(α, N, γ)
    rawTimeY = cumsum(jumpsY)

    tY = rawTimeY*T / maximum(rawTimeY)
    x = cumsum(symmetric_alpha_levy(α, N, γ^(α/2)))
    y = cumsum(symmetric_alpha_levy(α, N, γ^(α/2)))
    tOut = collect(0:1:N).*T/N
    xOut = Vector{Float64}(undef, N)
    yOut = Vector{Float64}(undef, N)

    for n in 1:N
        xOut[n] = x[find_nearest(tX, tOut[n]), 1]
        yOut[n] = y[find_nearest(tY, tOut[n]), 1]
    end
    return xOut, yOut, tOut
end

"""
Permute the states of `hmm` according to `perm`.

**Arguments**

- `perm::Vector{<:Integer}`: permutation of the states.

**Example**
```julia
using Distributions, HMMBase
hmm = HMM([0.8 0.2; 0.1 0.9], [Normal(0,1), Normal(10,1)])
hmm = permute(hmm, [2, 1])
hmm.A # [0.9 0.1; 0.2 0.8]
hmm.B # [Normal(10,1), Normal(0,1)]
```
"""