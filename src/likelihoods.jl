function _likelihood!(dR, D, L, dt, er)
    d = Diffusion.(D, dt, er)
    @inbounds for i = 1:track_num
        @inbounds for t = 1:track_length[i]+1
            @inbounds for s = 1:K
                L[t, s, i] = pdf.(d[s], ifelse(dR[t, s, i] < 1e-4, 1e-4, dR[t, s, i]))
            end
        end
    end
end

# function likelihood!(
#     observations::AbstractArray,
#     L::AbstractArray,
#     D,
#     dt,
#     er,
#     track_length,
# )
#     @argcheck size(observations) == size(L)
#     K, N = size(L, 2), size(L, 3)
#     d = Diffusion.(D, dt, er)
#     @inbounds for i = 1:N
#         @inbounds for t = 1:track_length[i]+1
#             @inbounds for j = 1:K
#                 L[t, j, i] =
#                     pdf.(
#                         d[j],
#                         ifelse(observations[t, j, i] < 1e-4, 1e-4, observations[t, j, i]),
#                     )
#             end
#         end
#     end
# end

function loglikelihood!(observations, L, d)
    @argcheck size(observations, 1) == size(L, 1)
    @argcheck size(observations, 2) == size(L, 3)
    K, N = size(L, 2), size(L, 3)
    fill!(L, nothing)
    @inbounds for n = 1:N
        T = length(filter(!isnothing, observations[:, n]))
        for t = 1:T
            for k = 1:K
                L[t, k, n] = pdf(d[k], observations[t, n])
            end
        end
    end
end
