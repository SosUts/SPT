function add_dR!(df::DataFrame)
    df[!, :dX] .= prepend!(diff(df.POSITION_X), NaN)
    df[!, :dY] .= prepend!(diff(df.POSITION_Y), NaN)
    df[!, :corrected_dX] .= prepend!(diff(df.corrected_x), NaN)
    df[!, :corrected_dY] .= prepend!(diff(df.corrected_y), NaN)
    df[df.New_Frame.==1, [:dX, :dY, :corrected_dX, :corrected_dY]] .= NaN
    df.dR2 = abs2.(df.dX) + abs2.(df.dY)
    df.dR = sqrt.(df.dR2)
    df.corrected_dR2 = abs2.(df.corrected_dX) + abs2.(df.corrected_dY)
    df.corrected_dR = sqrt.(df.corrected_dR2)
end

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

function displacement(
    df::DataFrame,
    id::Symbol = :TrackID,
    x::Symbol = :POSITION_X,
    y::Symbol = :POSITION_Y;
    δ::Int = 1,
)
    dr = Float64[]
    @inbounds for n = 1:maximum(df[!, id])
        m = extract(df, n, id, [x, y])
        # m = Matrix(df[df[!, id].==n, [xlabel, ylabel]])
        @simd for t = 1:size(m, 1)-δ
            append!(dr, displacement(m, t, δ))
        end
    end
    dr
end

function data2matrix(
    df::DataFrame,
    track_num::Integer,
    max_length::Integer,
    track_length::AbstractArray,
    K::Integer,
    start_point::AbstractArray,
)
    data = zeros(Float64, (max_length, K, track_num))
    for i = 1:track_num
        for n = 0:track_length[i]-1
            data[n+1, :, i] .= df.dR[start_point[i]+n]
        end
    end
    data
end

function dr2matrix(df, id, data, frame)
    N = maximum(df[!, id])
    dr = Matrix{Union{Nothing,Float64}}(nothing, maximum(df[!, frame]), N)
    @inbounds for n = 1:N
        m = filter(!isnan, df[df[!, id] .== n, data])
        @simd for t = 1:length(m)
            dr[t, n] = m[t]
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

function create_prior(df::DataFrames.DataFrame, K::Integer, dt::Float64, er::Float64)
    a::Array{Float64,1} = rand(Float64, K)
    a /= sum(a)
    A::Array{Float64,2} = rand(Float64, (K, K))
    @inbounds for i = 1:K
        A[i, :] /= sum(A[i, :])
    end
    R = kmeans(filter(!isnan, abs2.(df.dR))', K, tol = 1e-6; maxiter = 10000)
    D = R.centers # get the cluster centers
    D /= 4dt
    D .- abs2(er) / dt
    D = reverse(sort(Array{Float64,1}(D[:])))
    return a, A, D
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