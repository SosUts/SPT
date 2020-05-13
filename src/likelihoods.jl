function likelihood!(
    obserbations::AbstractArray,
    d,
    L::AbstractArray
    )
    @argcheck size(obserbations) == size(L)
    K, N = size(L, 2), size(L, 3)
    fill!(L, 0.0)

    @inbounds for i in 1:N
        track_length = count(!iszero, obserbations[:,1,i])
        @inbounds for t in 1:track_length[i]
            @inbounds for j in 1:K
                L[t, j, i] = pdf.(
                    d[j],
                    ifelse(dR[t, j, i]<1e-4, 1e-4, dR[t, j, i])
                    )
            end
        end
    end
end

function loglikelihood!(
    obserbations::AbstractArray,
    d,
    L::AbstractArray
    )
    @argcheck size(obserbations) == size(L)
    K, N = size(L, 2), size(L, 3)
    fill!(L, 0.0)

    @inbounds for i in 1:N
        track_length = count(!iszero, obserbations[:,1,i])
        @inbounds for t in 1:track_length[i]
            @inbounds for j in 1:K
                L[t, j, i] = logpdf.(
                    d[j],
                    ifelse(dR[t, j, i]<1e-4, 1e-4, dR[t, j, i])
                    )
            end
        end
    end
end