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