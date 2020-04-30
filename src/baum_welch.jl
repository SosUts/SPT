function create_prior(K::Int64, dt::Float64, df::DataFrames.DataFrame, error::Float64)
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

function likelihood!(dR, D, L, dt, error)
    d = Diffusion.(D, dt, error)
    @inbounds for i in 1:tracknum
        @inbounds for t in 1:tracklength[i] + 1
            @inbounds for s in 1:K
                L[t, s, i] = pdf.(d[s], ifelse(dR[t, s, i] < 1e-4, 1e-4, dR[t, s, i]))
            end
        end
    end
end