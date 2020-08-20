function vacf(
        df::DataFrame,
        max_τ::Int,
        δ::Int,
    )
    N = maximum(df.TrackID)
    result = DataFrame(n = Int[], τ = Int[], value = Float64[])
    @inbounds for n in 1:N
        data = Matrix(df[df.TrackID .== n, [:x, :]])
        T = size(data, 1)
        @inbounds for τ in 0:max_τ
            s = 0.0
            count = 0
            @simd for t in 1:(T-δ-max_τ)
                vₜ = velocity(data, t, δ)
                v2 = velocity(data, t+τ, δ)
                s += c(vₜ, v2)
                count += 1
            end
            s /= count
            push!(result, [n, τ, s])
        end
    end
    result
end

velocity(a::AbstractArray, t::Int, δ::Int) = (a[t+δ, :] .- a[t, :]) ./ δ
c(v1::AbstractVector, v2::AbstractVector) = v1 ⋅ v2

# function vacf(
#         df::DataFrame,
#         max_τ::Int,
#         δ::Int,
#     )
#     result = DataFrame(n = Int[], τ = Int[], t = Int[], value = Float64[])
#     @inbounds for n in 1:maximum(df.TrackID)
#         data = Matrix(df[df.TrackID .== n, [:POSITION_X, :POSITION_Y]])
#         T = size(data, 1)
#         @inbounds for t in 1:(T-δ-max_τ)
#             @inbounds for τ in 0:max_τ
#                 vₜ = v(data, t, δ)
#                 v2 = v(data, t+τ, δ)
#                 push!(result, [n, τ, t, c(vₜ, v2)])
#             end
#         end
#         break
#     end
#     result
# end
