module spt

using ArgCheck
using Clustering
using CSV
using DataFrames
using Distributions
using HMMBase
using LinearAlgebra
using LsqFit
using Plots
using Statistics
using Glob
using LaTeXStrings
# using JuMP
# using Ipopt
using Flux
using Random
# using JuMP


import Random: AbstractRNG, GLOBAL_RNG
import StatsBase: sem
import Distributions: quantile, minimum, maximum, pdf, cdf, rand, logpdf, @check_args, @distr_support
import MLJBase:int

# exportしたい関数一覧
export
    # angle.jl
    moving_angle,
    spiff,
    anisotropy_delta_t,
    anisotropy_mean_displacement,
    # displacement.jl
    Diffusion,
    suffstats,
    fit_mle,
    # msd.jl
    mean_square_disaplcement,
    plot_msd,
    # messages.jl
    forward!,
    backward!,
    posterior!,
    update_ξ!,
    # mle.jl
    fit_baumwelch,
    # example_mle,
    # likelihoods.jl
    likelihood!,
    loglikelihood!,
    # utilities.jl
    create_prior,
    preproccsing!,
    data2matrix,
    nomapround,
    label_mean_displacement!,
    group_files,
    # viterbi.jl
    viterbi,
    viterbi!




# 読み込みたいjlファイル
include("angle.jl")
include("displacement.jl")
include("mle.jl")
include("msd.jl")
include("messages.jl")
include("viterbi.jl")
include("likelihoods.jl")
include("utilities.jl")

# To be removed in a future version
# ---------------------------------
# export
    # n_parameters,

# @deprecate n_parameters(hmm) nparams(hmm)

end # module
