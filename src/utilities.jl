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

nomapround(b, d) = (x -> round.(x, digits=d)).(b)