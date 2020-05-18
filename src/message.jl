function forward!(
        α::AbstractArray,
        c::AbstractArray,
        a::AbstractArray,
        A::AbstractArray,
        L::AbstractArray,
        i::Integer, track_length
    )
    T, K, N = size(L)
    (T == 0) && return
    track_length = count(!iszero, L[:,1,i])
    α[1, :, i] = a .* L[1, :, i]
    c[1, i] = sum(α[1, :, i])
    α[1, :, i] /= c[1, i]
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

function backward!(
        β::AbstractArray,
        c::AbstractArray,
        A::AbstractArray,
        L::AbstractArray,
        i::Integer, track_length
    )
    T, K, N = size(L)
    @inbounds for j in 1:K
        β[track_length[i]+1, j, i] = 1.0
    end
    @inbounds for t in reverse(1:track_length[i])
        @inbounds for j1 in 1:K
            @inbounds @simd for j2 in 1:K
                β[t, j1, i] += β[t+1, j2, i] * A[j1, j2] * L[t+1, j2, i]
            end
        end
        @inbounds for j in 1:K
            β[t,j,i] /= c[t+1, i]
        end
    end
end

function posterior!(
        γ::AbstractArray,
        α::AbstractArray,
        β::AbstractArray,
        L::AbstractArray,
        i::Integer
    )
    @argcheck size(α) == size(β)
    T, K, N = size(L)
    track_length = count(!iszero, L[:,1,i])
    @inbounds for t in 1:track_length
        @inbounds @simd for j in 1:K
            γ[t,j,i] = α[t,j,i] * β[t,j,i]
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
        i::Integer
    )
    @argcheck size(α) == size(β)
    T, K, N = size(L)
    track_length = count(!iszero, L[:,1,i])
    @inbounds for t in 1:track_length-1
        @inbounds for j1 in 1:K
            @inbounds @simd for j2 in 1:K
                ξ[t,j1,j2,i] += (α[t,j1,i] * A[j1,j2] * L[t+1,j2,i]) / c[t+1, i]
            end
        end
    end
end