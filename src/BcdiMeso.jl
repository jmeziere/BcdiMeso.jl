module BcdiMeso
    using CUDA
    using Optim
    using LineSearches
    using BcdiCore
    using StatsBase

    include("State.jl")
    include("Operators.jl")
end
