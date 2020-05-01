function data2matrix(
        df::DataFrame, track_num::Integer, max_length::Integer,
        track_length::AbstractArray, K::Integer, start_point::AbstractArray,
    )
    data = zeros(Float64, (max_length, K, track_num))
    for i in 1:track_num
        for n in 1:track_length[i]+1
            data[n,:,i] .= df.dR[start_point[i]+n]
        end
    end
    data
end

function create_prior(K::Integer, dt::Float64, df::DataFrames.DataFrame, error::Float64)
    a::Array{Float64,1} = rand(Float64, K)
    a /= sum(a)
    A::Array{Float64,2} = rand(Float64, (K, K))
    @inbounds for i in 1:K
        A[i, :] /= sum(A[i, :])
    end
    R = kmeans(filter(!isnan, abs2.(df.dR))', K, tol = 1e-6 ; maxiter = 10000)
    D = R.centers # get the cluster centers
    D /= 4dt
    D .- abs2(error) / dt
    D = reverse(sort(Array{Float64,1}(D[:])))
    return a, A, D
end

function likelihood!(dR, D, L, dt, error)
    d = Diffusion.(D, dt, error)
    @inbounds for i in 1:tracknum
        @inbounds for t in 1:tracklength[i] + 1
            @inbounds for s in 1:K
                L[t, s, i] = pdf.(d[s], ifelse(dR[t, s, i] < 1e-4, 1e-4, dR[t, s, i]))
            end
        end
    end
end

function rand(
        rng::AbstractRNG, hmm::AbstractHMM{Univariate},
        z::AbstractArray{<:Integer}, T::Integer,
        N::Integer;
    )
    y = Array{Float64}(undef, size(z, 1), size(z, 2))
    for n in 1:N
        for t in 1:T
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