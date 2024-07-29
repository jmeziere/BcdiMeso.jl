module BcdiMeso
    using CUDA
    using Optim
    using LineSearches
    using BcdiCore

    include("State.jl")
    include("Operators.jl")
end
