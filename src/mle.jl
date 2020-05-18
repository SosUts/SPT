function fit_baumwelch(
        observations::AbstractArray,
        track_length::AbstractArray;
        tol = 1e-4,
        maxiter = 1e4
    )
    result = DataFrame(
        Iteration = Int64[],
        likelihood = Float64[],
        a = [],
        D = [],
        A = []
        )
    @argcheck size(observations, 1) != 0
    @argcheck maxiter >= 0

#     history = EMHistory2(false, 0, [], [], [], [])
    iteration::Integer = 0
    l::Float64 = 0.0
    ϵ::Float64 = 100.0
    likelihood = Float64[]

    T, K, N = size(observations)
    a, A, D = create_prior(K, dt, df, error)
#     println(a, D)
    α = zeros(Float64, (T, K, N))
    β = zeros(Float64, (T, K, N))
    γ = zeros(Float64, (T, K, N))
    ξ = zeros(Float64, (T, K, K, N))
    c = zeros(Float64, (T, N))
    L = zeros(Float64, (T, K, N))

    while ϵ > 1e-4
        pl = l
        fill!(α, 0.0)
        fill!(β, 0.0)
        fill!(γ, 0.0)
        fill!(ξ, 0.0)
        fill!(c, 0.0)
        fill!(L, 0.0)

        d = Diffusion.(D, dt, error)
        likelihood!(observations, L, D, dt, error, track_length)

        @inbounds for i in 1:size(observations, 3)
            forward!(α, c, a, A, L, i, track_length)
            backward!(β, c, A, L, i, track_length)
            posterior!(γ, α, β, L ,i)
            update_ξ!(ξ, α, β, c, A, L, i)
        end
        a = sum(γ[1, :, :], dims=2)[:, 1] ./ sum(γ[1, :, :])
        D = reshape(
            sum(abs2.(observations) .* γ , dims=(1,3)) ./ (sum(γ, dims=(1,3)) .* (4dt)),
            K)
        D .-= error^2/dt
        println(D)
        A = reshape(sum(ξ, dims=(1, 4)), K,K) ./ sum(reshape(sum(ξ, dims=(1, 4)), K,K), dims=2)
        iteration += 1
        replace!(log.(c), -Inf=>0.0)
        l = sum(c)
        ϵ = abs(l-pl)
        if iteration % 50 == 0
            println("iteration = $iteration")
            println("a = $a")
            println("D = $D")
            println("likelihood = $ϵ")
        end
        push!(result,(
                iteration, l, a, D, A
            ))
    end
#     history.converged = true
    println("EM converged in $iteration loops, ϵ = $ϵ")
    return result
end