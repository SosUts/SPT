# LinearAlgebra.cross couldn't be applied to 2d vector
function twod_cross(x::AbstractVector, y::AbstractVector)
    x[1] * y[2] - x[2] * y[1]
end

function moving_angle(x::AbstractVector, y::AbstractVector)
    cos_theta = dot(x, y) / (norm(x) * norm(y))
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
        if frac_x[i] == 0
            continue
        end
        if frac_y[i] == 0
            continue
        end
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
