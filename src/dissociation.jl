function dissociation_calculation(df::DataFrame)
    result = DataFrame(
        delta_t = Float64[],
        dissociation_rate = Float64[]
    )
    tmp_df = tmp_df = df[df.frame_num.==2, :]
    for i in 1:maximum(df.lifetime)
        delta_t = i*(1/45)
        rate = nrow(tmp_df[tmp_df.lifetime.==i, :])
        push!(result, [delta_t, rate])
    end
    result[:, :rate] ./= maximum(result[:, :dissociation_rate])
end

# function fit_dissociation(
#     df::DataFrame
#     )
#     model(x, p)
# end

# function fit_msd(
#     df::DataFrame;
#     max_time::Int64 = 10,
#     loc_error::Float64 = 0.03,
#     p0::AbstractVector = [1.0, 1.0]
#     )
#     @. model(x, p) = 4 * p[1] * x^p[2] + 4 * 0.03^2
#     fit = curve_fit(model, df.delta_t[1:max_time], df.msd[1:max_time], p0)
#     fit.param
# end