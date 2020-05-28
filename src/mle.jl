function fit_baumwelch(
    df::DataFrames.DataFrame;
    K::Integer,
    tol = 1e-2,
    maxiter = 2000,
    dt::Float64 = 0.022,
    er::Float64 = 0.03
    )
    result = DataFrame(
        iteration = Int64[],
        loglikelihood = Float64[],
        a = [],
        D = [],
        A = []
        )
    @argcheck maxiter >= 0

    iteration::Integer = 0
    l::Float64 = 0.0
    ϵ::Float64 = 100.0
    likelihood = Float64[]
    track_length, track_num, max_length, start_point = preproccsing!(df);
    observations = data2matrix(
        df, track_num, max_length, track_length,
        K, start_point
        )
    T, K, N = size(observations)
    a, A, D = create_prior(df, K, dt, er)
    α = zeros(Float64, (T, K, N))
    β = zeros(Float64, (T, K, N))
    γ = zeros(Float64, (T, K, N))
    ξ = zeros(Float64, (T, K, K, N))
    c = zeros(Float64, (T, N))
    L = zeros(Float64, (T, K, N))

    while ϵ > tol
        pl = l
        fill!(α, 0.0)
        fill!(β, 0.0)
        fill!(γ, 0.0)
        fill!(ξ, 0.0)
        fill!(c, 0.0)
        fill!(L, 0.0)

        d = Diffusion.(D, dt, er)
        likelihood!(observations, L, D, dt, er, track_length)

        @inbounds for i in 1:size(observations, 3)
            forward!(α, c, a, A, L, i, track_length)
            backward!(β, c, A, L, i, track_length)
            posterior!(γ, α, β, L ,i, track_length)
            update_ξ!(ξ, α, β, c, A, L, i, track_length)
        end
        a = sum(γ[1, :, :], dims=2)[:, 1] ./ sum(γ[1, :, :])
        D = reshape(
            sum(abs2.(observations) .* γ , dims=(1,3)) ./ (sum(γ, dims=(1,3)) .* (4dt)),
            K)
        D .-= er^2/dt

        println("D = $D")
        println(any(x -> x <= 0, D))
        if any(x -> x <= 0, D)
            println("D <= 0")
            for j in 1:K
                if D[j] <= 0
                    D[j] =
                    example_mle(
                        observations, track_length, D, γ, dt, er, i, j
                        )
                end
            end
            break
        end
        A = reshape(sum(ξ, dims=(1, 4)), K,K) ./ sum(reshape(sum(ξ, dims=(1, 4)), K,K), dims=2)

        # Note : filter was ~ 2times faster than replace
        l = sum(filter(!isinf, log.(c)))
        # l = sum(replace(log.(c), -Inf=>0.0))

        ϵ = abs(l-pl)
        if iteration % 100 == 0
            println("iteration = $iteration")
            println("a = $a")
            println("D = $D")
            println("loglikelihood = $ϵ")
        end
        push!(result,(
                iteration, l, a, D, A
            ))
        iteration += 1
    end
    aic = -2*l + 2*(2K-1)
    bic = -2*l + (2K-1)*log(sum(track_length .+ 1))
    if iteration < maxiter
        println("EM converged in $iteration loops, ϵ = $ϵ")
        println("#---------------------#")
        println("a = $a")
        println("D = $D")
        println("AIC = $aic")
        println("BIC = $bic")
    else
        println("EM DID NOT converged in $iteration loops, ϵ = $ϵ")
    end
    return result
end

function example_mle(
    observations, track_length, D, γ, dt, er, i, j
    )
    n = track_length[i]
    Random.seed!(1234)
    model = Model(Ipopt.Optimizer)
    @variable(model, 10.0 >= D >= 0.0, start = 1.0)
    @NLobjective(model, Max, sum(γ[x, j, i] for x in 1:n) *
        (sum((log(observations[x, j, i]) for x in 1:n)) - n * log(2(dt*D+er^2)) -
        (sum((observations[x, j, i])^2 for x in 1:n) / 4(dt*D + er^2)))
    );
    JuMP.optimize!(model);
    JuMP.value(D);
end