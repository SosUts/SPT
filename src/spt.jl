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
import Distributions: quantile, minimum, maximum, pdf, cdf, @check_args

# exportしたい関数一覧
export
    # msd.jl
    # baum_welch.jl
    # viterbi.jl
    # angle.jl
    # utilities.jl

# 読み込みたいjlファイル
include("hmm.jl")
include("mle.jl")
include("mle_api.jl")
include("mle_init.jl")
include("messages.jl")
include("messages_api.jl")
include("viterbi.jl")
include("viterbi_api.jl")
include("likelihoods.jl")
include("likelihoods_api.jl")
include("utilities.jl")

# To be removed in a future version
# ---------------------------------
export
    n_parameters,

# @deprecate n_parameters(hmm) nparams(hmm)

end # module
