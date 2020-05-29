function viterbi(
        a::AbstractVector,
        A::AbstractArray,
        L::AbstractArray,
        track_length::AbstractArray;
        logl = false
    )
    T1 = Array{Float64}(undef, size(L))
    T2 = Array{Int}(undef, size(L))
    z = Matrix{Int}(undef, size(L, 1), size(L, 3))
    c = Matrix{Float64}(undef, size(z))
    if logl
        error("logviterbi! not implemented yet.")
#         viterbilog!(T1, T2, z, a, A, L)
    else
        warn_logl(L)
        viterbi!(T1, T2, z, a, A, L, c, track_length)
    end
    z
end

function viterbi!(
        T1::AbstractArray,
        T2::AbstractArray,
        z::AbstractArray,
        a::AbstractVector,
        A::AbstractArray,
        L::AbstractArray,
        c::AbstractArray,
        track_length::AbstractArray
    )
    @argcheck size(T1, 1) == size(T2, 1) == size(L, 1) == size(z, 1) == size(c, 1)
    @argcheck size(T1, 2) == size(T2, 2) == size(L, 2) == size(a, 1) == size(A, 1) == size(A, 2)
    @argcheck size(T1, 3) == size(T2, 3) == size(L, 3) == size(z, 2) == size(c, 2)

    T, K, N = size(L)
    (T == 0) && return

    fill!(T1, 0.0)
    fill!(T2, 0)
    fill!(c, 0.0)

    for i in 1:N

        for j in 1:K
            T1[1, j, i] = a[j] * L[1, j, i]
            c[1, i] += T1[1, j, i]
        end

        for j in 1:K
            T1[1, j, i] /= c[1, i]
        end

        @inbounds for t = 2:track_length[i]
            for j1 in 1:K
                # TODO: If there is NaNs in T1 this may
                # stay to 0 (NaN > -Inf == false).
                # Hence it will crash when computing z[t].
                # Maybe we should check for NaNs beforehand ?
                amax = 0
                vmax = -Inf

                for j2 in 1:K
                    v = T1[t-1, j2, i] * A[j2, j1]
                    if v > vmax
                        amax = j2
                        vmax = v
                    end
                end

                T1[t, j1, i] = vmax * L[t, j1, i]
                T2[t, j1, i] = amax
                c[t, i] += T1[t, j1, i]
            end

            for j in 1:K
                T1[t, j, i] /= c[t, i]
            end
        end
    end

    z[track_length[i], i] = argmax(T1[track_length[i], :, i])
    for t in reverse(1:track_length[i]-1)
        z[t, i] = T2[t+1, z[t+1, i], i]
    end
end