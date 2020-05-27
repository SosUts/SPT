function preproccsing!(df::DataFrames.DataFrame)
    df[!, :dX] .= 0.0
    df[!, :dY] .= 0.0
    df.dX[2:end] .= diff(df.corrected_x)
    df.dY[2:end] .= diff(df.corrected_y)
    df[df.New_Frame .== 0, [:dX, :dY]] .= NaN
    df.dR2 = abs2.(df.dX) + abs2.(df.dY)
    df.dR = sqrt.(df.dR2)
    start_point = findall(x -> x == 1, df.New_Frame)
    end_point = start_point[2:end] .- 2
    terminus = size(df, 1)
    append!(end_point, terminus)
    track_length = Integer[]
    track_length = end_point .- start_point .+ 1
    track_num = maximum(df.TrackID)
    max_length = Int64(maximum(df.New_Frame))

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
        for n = 0:track_length[i]-1
            data[n+1, :, i] .= df.dR[start_point[i]+n]
        end
    end
    data
end

function create_prior(df::DataFrames.DataFrame, K::Integer, dt::Float64, er::Float64)
    a::Array{Float64,1} = rand(Float64, K)
    a /= sum(a)
    A::Array{Float64,2} = rand(Float64, (K, K))
    @inbounds for i = 1:K
        A[i, :] /= sum(A[i, :])
    end
    R = kmeans(filter(!isnan, abs2.(df.dR))', K, tol = 1e-6; maxiter = 10000)
    D = R.centers # get the cluster centers
    D /= 4dt
    D .- abs2(er) / dt
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

function group_files(dir::AbstractString)
    cd(dir)
    genotypes = readdir()
    filter!(isdir, genotypes)
    df = []
    for genotype in genotypes
        files = glob("./$genotype/*.csv")
        for file in files
            tmp_df = CSV.read(file)
            tmp_df[:, :Genotype] .= genotype
            push!(df, tmp_df)
        end
    end
    df = vcat(df...)
    grouped_df = groupby(df, [:Genotype])
    return grouped_df
end