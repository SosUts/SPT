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