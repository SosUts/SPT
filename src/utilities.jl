function preproccsing!(df::DataFrames.DataFrame)
    df[!, :dX] .= 0.0
    df[!, :dY] .= 0.0
    df.dX[2:end] .= diff(df.corrected_x)
    df.dY[2:end] .= diff(df.corrected_y)
    df[df.FRAME.==0, [:dX, :dY]] .= NaN
    df.dR2 = abs2.(df.dX) + abs2.(df.dY)
    df.dR = sqrt.(df.dR2)

    start_point = findall(x -> x == 1, df.New_Frame)
    endpoint = start_point[2:end] .- 2
    terminus = size(df)[1] - 1
    endpoint = append!(endpoint, terminus)
    track_length = endpoint .- start_point
    track_num = maximum(df.TrackID)
    max_length = maximum(df.New_Frame)

    return track_length, track_num, max_length, start_point
end

function data2matrix(
    df::DataFrame,
    track_num::Integer,
    max_length::Integer,
    track_length::AbstractArray,
    K::Integer,
    start_point::AbstractArray,
)
    data = zeros(Float64, (max_length, K, track_num))
    for i = 1:track_num
        for n = 1:track_length[i]+1
            data[n, :, i] .= df.dR[start_point[i]+n]
        end
    end
    data
end

function create_prior(
    df::DataFrames.DataFrame,
    K::Integer,
    dt::Float64,
    error::Float64
    )
    a::Array{Float64,1} = rand(Float64, K)
    a /= sum(a)
    A::Array{Float64,2} = rand(Float64, (K, K))
    @inbounds for i = 1:K
        A[i, :] /= sum(A[i, :])
    end
    R = kmeans(filter(!isnan, abs2.(df.dR))', K, tol = 1e-6; maxiter = 10000)
    D = R.centers # get the cluster centers
    D /= 4dt
    D .- abs2(error) / dt
    D = reverse(sort(Array{Float64,1}(D[:])))
    return a, A, D
end

nomapround(b, d) = (x -> round.(x, digits = d)).(b)


# function rand(
#     rng::AbstractRNG,
#     hmm::AbstractHMM{Univariate},
#     z::AbstractArray{<:Integer},
#     T::Integer,
#     N::Integer
# )
#     y = Array{Float64}(undef, size(z, 1), size(z, 2))
#     for n = 1:N
#         for t = 1:T
#             y[t, n] = rand(rng, hmm.B[z[t, n]])
#         end
#     end
#     y
# end

# function rand(
#     rng::AbstractRNG,
#     hmm::AbstractHMM,
#     T::Integer,
#     N::Integer;
#     init = rand(rng, Categorical(hmm.a), N),
#     seq = true,
# )
#     z = Matrix{Int}(undef, T, N)
#     for n = 1:N
#         (T >= 1) && (N >= 1) && (z[1, n] = init[n])
#         for t = 2:T
#             z[t, n] = rand(rng, Categorical(hmm.A[z[t-1, n], :]))
#         end
#     end
#     y = rand(rng, hmm, z, T, N)
#     seq ? (z, y) : y
# end

# rand(hmm::AbstractHMM, T::Integer, N::Integer; kwargs...) =
#     rand(GLOBAL_RNG, hmm, T, N; kwargs...)

# rand(hmm::AbstractHMM, z::AbstractArray{<:Integer}) =
#     rand(GLOBAL_RNG, hmm, size(z, 1), size(z, 2))

