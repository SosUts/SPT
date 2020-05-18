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
            y[t, n] = rand(rng, hmm.B[z[t, n]])
        end
    end
    y
end

function rand(
    rng::AbstractRNG,
    hmm::AbstractHMM,
    T::Integer,
    N::Integer;
    init = rand(rng, Categorical(hmm.a), N),
    seq = true,
)
    z = Matrix{Int}(undef, T, N)
    for n = 1:N
        (T >= 1) && (N >= 1) && (z[1, n] = init[n])
        for t = 2:T
            z[t, n] = rand(rng, Categorical(hmm.A[z[t-1, n], :]))
        end
    end
    y = rand(rng, hmm, z, T, N)
    seq ? (z, y) : y
end

rand(hmm::AbstractHMM, T::Integer, N::Integer; kwargs...) =
    rand(GLOBAL_RNG, hmm, T, N; kwargs...)

rand(hmm::AbstractHMM, z::AbstractArray{<:Integer}) =
    rand(GLOBAL_RNG, hmm, size(z, 1), size(z, 2))
