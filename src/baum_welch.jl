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

function forward!(
        α::AbstractArray,
        c::AbstractArray,
        a::AbstractArray,
        A::AbstractArray,
        L::AbstractArray,
    )
    fill!(L, 0.0)
    likelihood!(dR, D, L, dt, error)
    T, K, N = size(L)
    (T == 0) && return

    fill!(α, 0.0)
    fill!(c, 0.0)

    @inbounds for i in 1:N
        track_length = count(!iszero, L[:,1,i])
        @inbounds for j in 1:K
            α[1, j, i] = a[j] * L[1, j, i]
            c[1, i] += α[1, j, i]
            α[1, j, i] /= c[1, i]
        end
        @inbounds for t in 2:track_length
            @inbounds for j2 in 1:K
                @inbounds @simd for j1 in 1:K
                    α[t,j2,i] += α[t-1,j1,i] * A[j1, j2]
                end
            end
            @inbounds @simd for j in 1:K
                α[t,j,i] *= L[t,j,i]
                c[t,i] += α[t,j,i]
            end
            @inbounds @simd for j in 1:K
                α[t,j,i] /= c[t,i]
            end
        end
    end
    return c
end

function backward!(
        β::AbstractArray,
        c::AbstractArray,
        a::AbstractArray,
        A::AbstractArray,
        L::AbstractArray,
    )
    T, K, N = size(L)
    (T == 0) && return

    fill!(β, 0.0)
    @inbounds for i in 1:N
        track_length = count(!iszero, L[:,1,i])
        @inbounds for j in 1:K
            β[track_length, j, i] = 1.0
        end
        @inbounds for t in reverse(1:track_length-1)
            @inbounds for j1 in 1:K
                @inbounds @simd j2 in 1:K
                β[t, j1, i] += β[t+1, j2, i] * A[j1, j2] * L[t+1, j2, i]
            end
            for j in 1:K
                β[t,j,i] /= c[t+1]
            end
        end
    end
end

function posterior!(
        γ::AbstractArray,
        α::AbstractArray,
        β::AbstractArray,
        L::AbstractArray,
    )
    @argcheck size(α) == size(β)
    T, K, N = size(L)

    fill!(γ, 0.0)
    @inbounds for i in 1:N
        track_length = count(!iszero, L[:,1,i])
        @inbounds for t in 1:track_length
            @inbounds @simd for j in 1:K
                γ[t,s,i] = α[t,s,i] * β[t,s,i]
            end
        end
    end
end