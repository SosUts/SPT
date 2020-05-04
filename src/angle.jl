using LinearAlgebra

# LinearAlgebra.cross couldn't be applied to 2d vector
function twod_cross(
    x::AbstractVector,
    y::AbstractVector
    )
    x[1]*y[2]-x[2]*y[1]
end

function moving_angle(
        x::AbstractVector,
        y::AbstractVector
    )
    cos_theta = (x ⋅ y) / (norm(x)*norm(y))
    theta = rad2deg(acos(cos_theta))
    theta = ifelse(twod_cross(x,y)<0, -theta, theta)
    return theta
end

function spiff(
        x::AbstractArray,
        y::AbstractArray
    )
    @argcheck length(x) == length(y)
    total_len = length(x)

    spiff_x = [modf(i)[1] for i in x]
    spiff_x_ceil = [modf(i)[2] for i in x]
    spiff_y = [modf(i)[1] for i in y]
    spiff_y_ceil = [modf(i)[2] for i in y]

    x_plus = sort(spiff_x[spiff_x .>= 0.5])
    x_minus = sort(spiff_x[spiff_x .< 0.5])
    y_plus = sort(spiff_y[spiff_y .>= 0.5])
    y_minus = sort(spiff_y[spiff_y .< 0.5])

    len_x_plus, len_x_minus = length(x_plus), length(x_minus)
    len_y_plus, len_y_minus = length(y_plus), length(y_minus)

    corrected_x = zero(spiff_x)
    corrected_y = zero(spiff_y)
    @inbounds for i in 1:total_len
        if spiff_x[i] == 0
            continue
        end
        x1_curr = x[i]
        y1_curr = y[i]
        spiff_x_curr = spiff_x[i]
        spiff_y_curr = spiff_y[i]
        if spiff_x_curr >= 0.5
            tmp = sum(x_plus .<= spiff_x_curr)/len_x_plus
            corrected_x[i] = ceil(x1_curr) + tmp/2
        else
            tmp = sum((x_minus) .> (spiff_x_curr))/len_x_minus
            corrected_x[i] = ceil(x1_curr) - tmp/2
        end
        if spiff_y_curr >= 0.5
            tmp = sum(y_plus .<= spiff_y_curr)/len_y_plus
            corrected_y[i] = ceil(y1_curr) + tmp/2
        else
            tmp = sum((y_minus) .> (spiff_y_curr))/len_y_minus
            corrected_y[i] = ceil(y1_curr) - tmp/2
        end
    end

    return corrected_x, corrected_y
end