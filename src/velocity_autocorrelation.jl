function vacf(
        df::DataFrame,
        xlabel::Symbol,
        ylabel::Symbol,
        idlabel::Symbol,
        max_τ::Int,
        δt::Int,
    )
    N = maximum(df.TrackID)
    result = DataFrame(n = Int[], τ = Int[], value = Float64[])
    @inbounds for n in 1:maximum(df[!, idlabel])
        data = Matrix(df[df[!, idlabel] .== n, [xlabel, ylabel]])
        T = size(data, 1)
        # maxt = max_τ
        # if T < max_τ
        #     maxt = T
        # end
        @inbounds for τ in 0:max_τ-δt
            s = 0.0
            @simd for t in 1:(T-δt-τ)
                vₜ = velocity(data, t, δt)
                v2 = velocity(data, t+τ, δt)
                s += c(vₜ, v2)
            end
            s /= (T-δt-τ)
            push!(result, [n, τ, s])
        end
    end
    result
end

velocity(r::AbstractArray, t::Int, δt::Int) = (r[t+δt, :] .- r[t, :]) ./ δt
c(v1::AbstractVector, v2::AbstractVector) = v1 ⋅ v2
