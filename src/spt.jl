module SPT

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

import Random: AbstractRNG, GLOBAL_RNG
import StatsBase: sem
import Distributions: quantile, minimum, maximum, pdf, cdf, rand, logpdf, @check_args, @distr_support

# exportしたい関数一覧
export
    # angle.jl
    twod_cross,
    moving_angle,
    spiff,
    # msd.jl
    Diffusion,
    mean_square_disaplcement,
    plot_msd,
    combine_msd_files,
    # messages.jl
    forward!,
    backward!,
    posterior!,
    update_ξ!,
    # mle.jl
    fit_baumwelch,
    # likelihoods.jl
    likelihood!,
    loglikelihood!,
    # utilities.jl
    create_prior,
    preproccsing!,
    data2matrix,
    nomapround

# 読み込みたいjlファイル
include("angle.jl")
include("displacement.jl")
include("mle.jl")
include("msd.jl")
include("messages.jl")
# include("viterbi_api.jl")
include("likelihoods.jl")
include("utilities.jl")

# To be removed in a future version
# ---------------------------------
# export
    # n_parameters,

# @deprecate n_parameters(hmm) nparams(hmm)

end # module
