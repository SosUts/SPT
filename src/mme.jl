function mean_maximal_excursion(
    r::AbstractMatrix,
    τ::Integer,
    k::Integer
    )
    @argcheck (k >= 1)&&(τ >= 0)
    @argcheck size(r, 1) >= 2
    T = size(r, 1)
    m = displacement(r, 1)
    @inbounds for δt in 1:τ
        mₜ = displacement(r, δt)
        if mₜ > m
            m = mₜ
        end
    end
    m^k
end

function calc_mme(
    df::DataFrame,
    idlabel::Symbol,
    xlabel::Symbol,
    ylabel::Symbol,
    )
    result = DataFrame(n = Int[], τ = Int[], m1 = Float64[])
    @inbounds for n in 1:maximum(df[!, idlabel])
        m = Matrix(df[df[!, idlabel].== n, [xlabel, ylabel]])
        @simd for τ in 1:size(m, 1)-1
            m1 = mean_maximal_excursion(m, τ, 1)
            push!(result, [n, τ, m1])
        end
    end
    result
end

function StatsBase.moment(
    df::DataFrame,
    idlabel::Symbol,
    xlabel::Symbol,
    ylabel::Symbol
)
    result = DataFrame(
        TrackID = Int64[],
        fourth_moment = Float64[],
        second_moment = Float64[],
        delta_t = Int64[],
        n = Int64[],
    )
    @inbounds for id in sort(collect(Set(df.TrackID)))
        data = convert(Matrix, df[df.TrackID.==id, [xlabel, ylabel]])
        T = size(data, 1)
        T = 100
        @inbounds for δ in 1:T-1
            c⁴ = 0.0
            c² = 0.0
            @inbounds for t in 1:(T-δ)
                c⁴ += spt.displacement(data, t, δ)^4
                c² += spt.displacement(data, t, δ)^2
            end
            c⁴ /= (T-δ)
            c² /= (T-δ)
            push!(result, [id, c⁴, c², δ, T-δ])
        end
    end
    result
end