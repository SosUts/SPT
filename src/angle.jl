# LinearAlgebra.cross couldn't be applied to 2d vector
function twod_cross(x::AbstractVector, y::AbstractVector)
    @argcheck length(x) == length(y) == 2
    x[1] * y[2] - x[2] * y[1]
end

function moving_angle(x::AbstractVector, y::AbstractVector)
    @argcheck length(x) == length(y) == 2
    cos_theta = dot(x, y) / (norm(x) * norm(y))
    cos_theta = round(cos_theta, digits=6)
    # if cos_theta > 1.0
    #     println(x, y)
    # end
    theta = rad2deg(acos(cos_theta))
    theta = ifelse(twod_cross(x, y) < 0, -theta, theta)
    theta
end

# Naive implemation is 2times faster than using ifelse function
function spiff(x::AbstractArray, y::AbstractArray)
    @argcheck length(x) == length(y)
    total_len = length(x)

    frac_x = [modf(i)[1] for i in x]
    int_x = [modf(i)[2] for i in x]
    frac_y = [modf(i)[1] for i in y]
    int_y = [modf(i)[2] for i in y]

    x_plus = sort(frac_x[frac_x.>=0.5])
    x_minus = sort(frac_x[frac_x.<0.5])
    y_plus = sort(frac_y[frac_y.>=0.5])
    y_minus = sort(frac_y[frac_y.<0.5])

    x_plus_len, x_minus_len = length(x_plus), length(x_minus)
    y_plus_len, y_minus_len = length(y_plus), length(y_minus)

    corrected_x = zero(frac_x)
    corrected_y = zero(frac_y)

    @inbounds for i = 1:total_len
        # if frac_x[i] == 0
        #     continue
        # end
        # if frac_y[i] == 0
        #     continue
        # end
        #         corrected_x[i] = ifelse(
        #             frac_x[i] >= 0.5,
        #             int_x[i] + (sum(x_plus .<= frac_x[i])/x_plus_len)/2,
        #             int_x[i] - (sum(x_minus .> frac_x[i])/x_minus_len)/2
        #         )
        #          corrected_y[i] = ifelse(
        #             frac_x[i] >= 0.5,
        #             int_y[i] + (sum(y_plus .<= frac_y[i])/y_plus_len)/2,
        #             int_y[i] - (sum(y_minus .> frac_y[i])/y_minus_len)/2
        #         )
        if frac_x[i] >= 0.5
            tmp = sum(x_plus .<= frac_x[i]) / x_plus_len
            corrected_x[i] = int_x[i] + tmp / 2
        else
            tmp = sum((x_minus) .> (frac_x[i])) / x_minus_len
            corrected_x[i] = int_x[i] - tmp / 2
        end
        if frac_y[i] >= 0.5
            tmp = sum(y_plus .<= frac_y[i]) / y_plus_len
            corrected_y[i] = int_y[i] + tmp / 2
        else
            tmp = sum((y_minus) .> (frac_y[i])) / y_minus_len
            corrected_y[i] = int_y[i] - tmp / 2
        end
    end
    return corrected_x, corrected_y
end

function anisotropy(
        df::DataFrames.DataFrame;
        maxt::Integer = 45,
        localization_error::Float64 = 0.03,
        num_of_resample = 50,
    )
    result = DataFrame(
        delta_t = Real[],
        cell_num = String[],
        fw = Real[],
        bw = Real[],
        n = Integer[],
        anisotropy = Float64[],
        std = Float64[],
        corrected_fw = Real[],
        corrected_bw = Real[],
        corrected_n = Integer[],
        corrected_anisotropy = Float64[],
        corrected_std = Float64[]
    )
    for t = 1:maxt
        tmp_data = df[
            (df.delta_t .== t) .& (df.dis_aft .>= localization_error) .& 
            (df.dis_bfr .>= localization_error), :relative_angle]
        tmp_data = convert(Array, tmp_data)
        corrected_tmp_data = df[
            (df.delta_t .== t) .& (df.corrected_dis_aft .>= localization_error) .&
            (df.corrected_dis_bfr .>= localization_error), :corrected_angle]
        corrected_tmp_data = convert(Array, corrected_tmp_data)
        cell_num = df[1, :cell_num]

        fw, bw, n, anisotropy = calculate_fw_bw(tmp_data)
        corrected_fw, corrected_bw, corrected_n, corrected_anisotropy = 
            calculate_fw_bw(corrected_tmp_data)

        boot_std = []
        corrected_boot_std = []
        for _ in 1:num_of_resample
            boot_df = tmp_data[
                rand(1:length(tmp_data), div(length(tmp_data), 2))]
            corrected_boot_df = corrected_tmp_data[
                rand(1:length(corrected_tmp_data), div(length(corrected_tmp_data), 2))]
            _, _, _, tmp_anisotropy = calculate_fw_bw(boot_df)
            _, _, _, corrected_tmp_anisotropy = calculate_fw_bw(corrected_boot_df)
            append!(boot_std, tmp_anisotropy)
            append!(corrected_boot_std, corrected_tmp_anisotropy)
        end

        push!(result, [
                t, cell_num, fw, bw, n, anisotropy, std(boot_std),
                corrected_fw, corrected_bw, corrected_n, corrected_anisotropy, std(corrected_boot_std)
                ])
    end
    return result
end

function calculate_fw_bw(data::AbstractArray)
    fw = size(data[-30 .<= data .<= 30, :], 1)
    bw = size(data[(150 .<= data) .|  (data .<= -150), :], 1)
    n = fw + bw
    anisotropy = bw / fw
    return fw, bw, n, anisotropy
end

function plot_anisotropy(grouped_df; maxt = 45, save_fig = false)
    for i = 1:length(grouped_df)
        result = []
        tmp_df = grouped_df[i]
        for t = 1:maxt
            a = tmp_df[tmp_df.delta_t.==t, :]
            push!(result, [t, mean(a.anisotropy), mean(a.std)])
        end

        result = reduce(vcat, result')
        genotype = tmp_df[1, :Genotype]
        plot(result[:, 1] * (1 / 45), result[:, 2], label = "$genotype")
        fill_between(
            result[:, 1] * (1 / 45),
            result[:, 2] .- result[:, 3],
            result[:, 2] .+ result[:, 3],
            alpha = 0.2,
        )
    end
    hlines([1], 0, 1, linestyle="--", color="k")

    xlim(0, maxt*(1/45))
    xlabel("Δt (sec)", fontsize = 14, family = "Arial")
    xticks(fontsize = 12)

    ylim(-1, )
    ylabel("Anisotropy", fontsize = 14, family = "Arial")
    yticks(fontsize = 12)

    legend(bbox_to_anchor = (1.02, 1.0), loc = 2, borderaxespad = 0.0, fontsize = 14)
    if save_fig
        savefig("anisotropy_dia0.25.png", bbox_inches = "tight", dpi = 800)
    end
end