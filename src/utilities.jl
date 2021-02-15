function preproccsing!(df::DataFrames.DataFrame)
    df[!, :dX] .= 0.0
    df[!, :dY] .= 0.0
    df.dX[2:end] .= diff(df.corrected_x)
    df.dY[2:end] .= diff(df.corrected_y)
    df[df.FRAME2.==1, [:dX, :dY]] .= NaN
    df.dR2 = abs2.(df.dX) + abs2.(df.dY)
    df.dR = sqrt.(df.dR2)
    start_point = findall(x -> x == 2, df.FRAME2)
    end_point = start_point[2:end] .- 2
    terminus = size(df, 1)
    append!(end_point, terminus)
    track_length = Integer[]
    track_length = end_point .- start_point .+ 1
    track_num = maximum(df.TrackID)
    max_length = Int64(maximum(df.FRAME2))

    return track_length, track_num, max_length, start_point
end

function chi(
    df;
    idlabel = :TrackID,
    datalabel = :POSITION_X,
    δ::Int = 1
)
    dx = Float64[]
    @inbounds for n = sort(collect(Set(df[!, idlabel])))
        m = extract(df, n, idlabel, [datalabel])
        @simd for t = 1:size(m, 1)-δ
            append!(dx, m[t+δ] - m[t])
        end
    end
    dx
end

function displacement(
    df;
    idlabel = :TrackID,
    xlabel = :POSITION_X,
    ylabel = :POSITION_Y,
    δ::Int = 1
)
    dr = Float64[]
    @inbounds for n = sort(collect(Set(df[!, idlabel])))
        m = extract(df, n, idlabel, [xlabel, ylabel])
        @simd for t = 1:size(m, 1)-δ
            append!(dr, displacement(m, t, δ))
        end
    end
    dr
end

function xy2matrix(df, id, x, y, frame)
    N = maximum(df[!, id])
    xy = Array{Union{Nothing,Float64}}(nothing, maximum(df[!, frame]), 2, N + 1 - minimum(df[!, id]))
    @inbounds for (n, i) = enumerate(minimum(df[!, id]):maximum(df[!, id]))
    # @inbounds for n = minimum(df[!, id]):N
        m = extract(df, i, id, [x, y])
        @simd for t = 1:size(m, 1)
            xy[t, 1, n] = m[t, 1]
            xy[t, 2, n] = m[t, 2]
        end
    end
    xy
end

nomapround(b, d) = (x -> round.(x, digits = d)).(b)

"""
    group_files(dir) -> GroupedDataFrame{DataFrame}

Make grouped dataframe that exist in dir.
# TODO: raise an error if no csv files exist in the directory.
"""
function group_files(dir::AbstractString)
    cd(dir)
    genotypes = readdir()
    filter!(isdir, genotypes)
    df = []
    for genotype in genotypes
        files = glob("./$genotype/*.csv")
        for file in files
            tmp_df = CSV.read(file)
            tmp_df[:, :Genotype] .= genotype
            push!(df, tmp_df)
        end
    end
    df = vcat(df...)
    grouped_df = groupby(df, [:Genotype])
    return grouped_df
end

function label_mean_displacement!(
    gd::GroupedDataFrame;
    displacement = :mean_dis,
    max_dis::Int = 1,
)
    @argcheck displacement in [:mean_dis, :corrected_mean_dis]
    @argcheck max_dis > 0
    bin_array = range(0, stop = max_dis, length = max_dis * 20)
    transform!(
        gd,
        :mean_dis => (x -> int(cut(x, bin_array, extend = true), type = Int)) => :bin,
        ungroup = false,
    )
    transform!(
        gd,
        :corrected_mean_dis =>
            (x -> int(cut(x, bin_array, extend = true), type = Int)) => :corrected_bin,
        ungroup = false,
    )
    return collect(bin_array)
end

function time_average(df::DataFrame, f::Function, id::Symbol, datalabel)
    result = []
    N = maximum(df[:, id])
    @inbounds for n = 1:N
        data = Matrix(df[df[!, id].==n, datalabel])
        T = size(data, 1)
        for δ = 1:T-1
            c = 0.0
            @simd for t = 1:T-δ
                c += f(data, t, δ)
            end
            c /= (T - δ)
            push!(result, [n, c, δ])
        end
    end
    result
end

"""
    add_noise!(m) -> Matrix
Update m by adding a random uniform noise(± 1e-4), if m[t, (1 or 2)] ≈ ,[t-1, (1 or 2)].
This is needed when diff(coordinates) == 0 cause an error.
```
"""
function add_noise!(m)
    @inbounds @simd for t = 2:size(m, 1)
        if m[t, 1] ≈ m[t-1, 1]
            m[t, 1] += rand(Uniform(-1e-4, 1e-4))
        end
        if m[t, 2] ≈ m[t-1, 2]
            m[t, 2] += rand(Uniform(-1e-4, 1e-4))
        end
    end
end

function extract(df::DataFrame, n::Int, id::Symbol, x::Symbol)
    Vector(@where(df, cols(id) .== n)[!, x])
    # Vector{Float64}(df[df[:, id] .== n, x])
end

function extract(df::DataFrame, n::Int, id::Symbol, label::Vector{Symbol})
    Matrix(@where(df, cols(id) .== n)[!, label])
end

# https://discourse.julialang.org/t/a-good-way-to-filter-an-array-with-keeping-its-dimension/42603
function remove_nothing(x::AbstractArray)
    x[.!any.(isnothing, eachrow(x)), :]
end

# function frameaverage(data; L = 2)
#     new_data = zero(size(data))
#     @inbounds for t = 1:L-1
#         new_data[t, ]
#     @inbounds for t = 1:size(data, 1)

# end

# function displacement(r::AbstractMatrix, δ::Int)
#     sqrt((r[1+δ, 2] - r[1, 2])^2 + (r[1+δ, 1] - r[1, 1])^2)
# end

# function displacement(r::AbstractMatrix, t::Int, δ::Int)
#     sqrt((r[t+δ, 2] - r[t, 2])^2 + (r[t+δ, 1] - r[t, 1])^2)
# end

# function squared_displacement(r::AbstractMatrix, t::Int, δ::Int)
#     displacement(r, t, δ)^2
# end