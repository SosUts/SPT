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

    iteration::Int = 0
    l::Float64 = 0.0
    ϵ::Float64 = 100.0
    track_length, track_num, max_length, start_point = preproccsing!(df)
    observations = dr2matrix(df, idlabel, datalabel, framelabel)
    a, A, D = create_prior(df, K, dt, er)

    c = Matrix{Union{Nothing, Float64}}(undef, T, N)
    α = Array{Union{Nothing, Float64}}(undef, T, K, N)
    β = Array{Union{Nothing, Float64}}(undef, T, K, N)
    γ = Array{Union{Nothing, Float64}}(undef, T, K, N)
    ξ = Array{Union{Nothing, Float64}}(undef, T, K, K, N)
    LL = Array{Union{Nothing, Float64}}(undef, T, K, N)

    while ϵ > tol
        pl = l
        fill!(α, nothing)
        fill!(β, nothing)
        fill!(γ, nothing)
        fill!(ξ, nothing)
        fill!(c, nothing)
        fill!(L, nothing)

        d = Diffusion.(D, dt, er)
        loglikelihoods!(L, observations, d)

        @inbounds for n in 1:N
            forward!(α, c, a, A, L, n)
            backward!(β, c, A, L, n, track_length)
            posterior!(γ, α, β, L ,n, track_length)
            update_ξ!(ξ, α, β, c, A, L, n, track_length)
        end
        a = sum(γ[1, :, :], dims=2)[:, 1] ./ sum(γ[1, :, :])
        D = reshape(
            sum(abs2.(observations) .* γ , dims=(1,3)) ./ (sum(γ, dims=(1,3)) .* (4dt)),
            K) .- er^2/dt
        # if any(x -> x <= 0, D)
        #     # println("D <= 0")
        #     for j in 1:K
        #         if D[j] <= 0
        #             D[j] = 1e-5
        # #             D[j] =
        # #             example_mle(
        # #                 observations, track_length, track_num, γ, dt, er, j
        # #                 )
        #         end
        #     end
        # end
        @show D
        A = reshape(sum(ξ, dims=(1, 4)), K,K) ./ sum(reshape(sum(ξ, dims=(1, 4)), K,K), dims=2)
        l = sum(filter(!isinf, log.(c)))

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

# function _objective(
#     γ::AbstractArray,
#     observations::AbstractArray,
#     D::Float64,
#     j::Integer,
#     )
#     γ_sum = sum(γ[:, j, ;])
#     γ_observations = sum(abs2.(observations[:, j, :]) .* γ[:, j, :])
#     γ_sum*log(dt*D+er^2) + γ_observations/4(dt*D+er^2)
# end

# function _update_D!(
#     grad::Float64,
#     D::Float64;
#     learning_rate::Float64
#     )
#     opt = RADAM()
#     D = param(D)
#     ps = params(model)
#     gs = gradient(ps) do
#         loss(x, y)
#     end

# Flux.Optimise.update!(opt, ps, gs)
#     @argcheck 0 <= learning_rate <= 1


# function example_mle(
#     observations, track_length, track_num, γ, dt, er, j
#     )
#     γ_sum = sum(γ[:, j, :])
#     model = Model(with_optimizer(Ipopt.Optimizer, print_level=0))
#     @variable(model, 1.0 >= D >= 0.0, start = 0.5)
#     @NLobjective(model, Max, γ_sum *
#         (sum((log(observations[t, j, n]) for n in 1:track_num for t in 1:track_length[n])) -
#         log(2(dt*D+er^2)) -
#         (sum(abs2.(observations[t, j, n]) for n in 1:track_num for t in 1:track_length[n]) / 4(dt*D + er^2)))
#     );
#     @NLobjective(model, Max, -γ_sum * log(2(dt*D+er^2)) -
#             sum(γ[t, j, n]*abs2(observations[t, j, n]) for n in 1:track_num for t in 1:track_length[n]) / 4(dt*D + er^2)
#                 )
#     @NLobjective(model, Max, γ_sum * sum(observations[t, j, n] for n in 1:track_num for t in 1:track_length[n]) -
#                     γ_sum*log(2(dt*D+er^2)) -
#                     sum(γ[t, j, n]*abs2(observations[t, j, n]) for n in 1:track_num for t in 1:track_length[n]) / 4(dt*D + er^2)
#                 )
#     JuMP.optimize!(model);
#     return value(D)
# end