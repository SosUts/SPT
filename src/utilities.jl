

function preproccsing!(df::DataFrames.DataFrame)
    df[!, :dX] .= 0.0
    df[!, :dY] .= 0.0
    df.dX[2:end] .= diff(df.spiff_x)
    df.dY[2:end] .= diff(df.spiff_y)
    df[df.FRAME.==0, [:dX, :dY]] .= NaN
    df.dR2 = abs2.(df.dX) + abs2.(df.dY)
    df.dR = sqrt.(df.dR2)

    startpoint = findall(x -> x == 0, df.FRAME)
    endpoint = startpoint[2:end] .- 2
    terminus = size(df)[1] - 1
    endpoint = append!(endpoint, terminus)
    track_length = endpoint .- startpoint
    track_num = maximum(df.TrackID)
    max_length = maximum(df.FRAME)

    return track_length, track_num, max_length
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
