function forward!(
    α::AbstractArray,
    c::AbstractArray,
    a::AbstractArray,
    A::AbstractArray,
    L::AbstractArray,
    n::Integer,
    track_length::AbstractArray,
)
    T, K, N = size(L)
    (T == 0) && return
    α[1, :, n] = a .* L[1, :, n]
    c[1, n] = sum(α[1, :, n])
    α[1, :, n] /= c[1, n]
    @inbounds for t = 2:track_length[n]
        @inbounds for j2 = 1:K
            @inbounds @simd for j1 = 1:K
                α[t, j2, n] += α[t-1, j1, n] * A[j1, j2]
            end
        end
        @inbounds @simd for j = 1:K
            α[t, j, n] *= L[t, j, n]
            c[t, n] += α[t, j, n]
        end
        @inbounds @simd for j = 1:K
            α[t, j, n] /= c[t, n]
        end
    end
end

function backward!(
    β::AbstractArray,
    c::AbstractArray,
    A::AbstractArray,
    L::AbstractArray,
    n::Integer,
    track_length::AbstractArray,
)
    T, K, N = size(L)
    @inbounds for j = 1:K
        β[track_length[n], j, n] = 1.0
    end
    @inbounds for t in reverse(1:track_length[n]-1)
        @inbounds for j1 = 1:K
            @inbounds @simd for j2 = 1:K
                β[t, j1, n] += β[t+1, j2, n] * A[j1, j2] * L[t+1, j2, n]
            end
        end
        @inbounds for j = 1:K
            β[t, j, n] /= c[t+1, n]
        end
    end
end

function posterior!(
    γ::AbstractArray,
    α::AbstractArray,
    β::AbstractArray,
    L::AbstractArray,
    n::Integer,
    track_length::AbstractArray
)
    @argcheck size(α) == size(β)
    T, K, N = size(L)
    @inbounds for t = 1:track_length[n]
        @inbounds @simd for j = 1:K
            γ[t, j, n] = α[t, j, n] * β[t, j, n]
        end
    end
end

function update_ξ!(
    ξ::AbstractArray,
    α::AbstractArray,
    β::AbstractArray,
    c::AbstractArray,
    A::AbstractMatrix,
    L::AbstractArray,
    n::Integer,
    track_length::AbstractArray
)
    @argcheck size(α) == size(β)
    T, K, N = size(L)
    @inbounds for t = 1:track_length[n]-1
        @inbounds for j1 = 1:K
            @inbounds @simd for j2 = 1:K
                ξ[t, j1, j2, n] += (α[t, j1, n] * A[j1, j2] * L[t+1, j2, n]) / c[t+1, n]
            end
        end
    end
end
