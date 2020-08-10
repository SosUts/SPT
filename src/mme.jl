function mme(r::AbstractMatrix, τ::Integer, k::Integer)
    @argcheck (k >= 1)&&(τ >= 0)
    @argcheck size(r, 1) >= 2
    T = size(r, 1)
    m = distance(r, 1)
    @inbounds for δt in 1:τ
        mₜ = distance(r, δt)
        if mₜ > m
            m = mₜ
        end
    end
    m^k
end

function calc_mme(df)
    result = DataFrame(n = Int[], τ = Int[], m1 = Float64[])
    @inbounds for n in 1:maximum(df.TrackID)
        m = Matrix(df[df.TrackID .== n, [:POSITION_X, :POSITION_Y]])
        @simd for τ in 1:size(m, 1)-1
            m1 = mme(m, τ, 1)
            push!(result, [n, τ, m1])
        end
    end
    result
end