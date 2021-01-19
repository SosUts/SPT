function add_dr!(
    df;
    idlabel = :TrackID,
    xlabel = :POSITION_X,
    ylabel = :POSITION_Y
)
    df[!, :dx] .= prepend!(diff(df[:, xlabel]), NaN)
    df[!, :dy] .= prepend!(diff(df[:, ylabel]), NaN)
    df.dr2 = abs2.(df.dx) + abs2.(df.dy)
    df.dr = sqrt.(df.dr2)

    @inbounds for i = 2:nrow(df)
        if df[i, idlabel] != df[i-1, idlabel]
            df[i, [:dx, :dy, :dr, :dr2]] .= NaN
        end
    end
end

function dr2matrix(df, idlabel, datalabel, framelabel)
    N = length(Set(df[!, idlabel]))
    dr = Matrix{Union{Nothing,Float64}}(nothing, maximum(df[!, framelabel])+1, N)
    @inbounds for (i, n) = enumerate(Set(df[!, idlabel]))
        m = filter(!isnan, df[df[!, idlabel] .== n, datalabel])
        @simd for t = 1:length(m)
            dr[t, i] = m[t]
        end
    end
    dr
end

function create_prior(df, K::Int, dt::Float64, er::Float64)
    a = rand(Float64, K)
    a /= sum(a)
    A = rand(Float64, K, K)
    @inbounds for i = 1:K
        A[i, :] /= sum(A[i, :])
    end
    R = kmeans(abs2.(filter(!isnothing, observations))', K, tol = 1e-6; maxiter = 10000)
    D = R.centers # get the cluster centers
    D /= 4dt
    D .- abs2(er) / dt
    D = reverse(sort(Array{Float64,1}(D[:])))
    a, A, D
end