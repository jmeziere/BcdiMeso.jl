abstract type Operator
end

struct OperatorList <: Operator
    ops::Vector{Operator}
end

function operate(ol::OperatorList, state::State)
    for op in ol.ops
        operate(op, state)
    end
end

"""
    OptimizeState(state, primitiveRecipLattice, numPeaks)

Create an object that performs an iteration of stochastic gradient descent.
`numPeaks` number of peaks are selected randomly. One step of gradient descent
is taken using the More-Thuente linesearch.

This implimentation takes into account effects of small angle measurement 
usually ignored in the BCDI problem. This is described in [carnis_towards_2019](@cite)
although this implimentation will be faster because a NUFFT is used instead of
many FFTs.
"""
struct MRBCDI{I,T1,T2} <: Operator
    numPeaks::Int64
    AInv::CuArray{Float64, 2, CUDA.Mem.DeviceBuffer}
    allU::CuArray{Float64, I, CUDA.Mem.DeviceBuffer}
    B::CuArray{Float64, 2, CUDA.Mem.DeviceBuffer}
    losses::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
    iterations::Int64
    ra0::Float64
    ua0::Float64
    alpha::Ref{Float64}
    TVRegs::T1
    BetaRegs::T2

    function MRBCDI(state, primitiveRecipLattice, numPeaks, iterations, lambdaTV, lambdaBeta, a, b, c, alpha)
        neighbors = CUDA.zeros(Int64, 6, length(state.keepInd))
        inds = [
            CartesianIndex(1,0,0), CartesianIndex(0,1,0), CartesianIndex(0,0,1),
            CartesianIndex(-1,0,0), CartesianIndex(0,-1,0), CartesianIndex(0,0,-1)
        ]
        function myFind(a,b)
            ret = findfirst(a,b)
            if ret == nothing
                return 0
            end
            return ret
        end
        function myMod(a,b)
            return CartesianIndex(mod(a[1]-1,b[1])+1, mod(a[2]-1,b[2])+1, mod(a[3]-1,b[3])+1)
        end
        s = CartesianIndex(size(state.cores[1].intens))
        for i in 1:length(inds)
            neighs = vec(myFind.(isequal.(myMod.(state.keepInd .- inds[i], s)), Ref(vec(state.keepInd))))
            neighbors[i,:] .= neighs
        end

        allU = CUDA.zeros(Float64, 4, size(state.keepInd)...)
        B = CUDA.zeros(Float64, 3, length(state.keepInd))
        losses = CUDA.zeros(Float64, length(state.cores))
        TVRegs = [BcdiCore.TVReg(
            lambdaTV * sqrt(reduce(+, state.cores[i].intens)) / reduce(+, state.support), neighbors
        ) for i in 1:length(state.cores)]
        BetaRegs = [BcdiCore.BetaReg(
            lambdaBeta * sqrt(reduce(+, state.cores[i].intens)) / reduce(+, state.support), a, b, c
        ) for i in 1:length(state.cores)]
        new{ndims(allU),typeof(TVRegs),typeof(BetaRegs)}(
            numPeaks, primitiveRecipLattice, allU, B, losses,
            iterations, 1, 500, alpha, TVRegs, BetaRegs
        )
    end
end

function operate(mrbcdi::MRBCDI, state)
    optimCore = StatsBase.sample(1:length(state.cores), mrbcdi.numPeaks, replace=false)
    ki = CartesianIndices(state.keepInd)
    iters = 0

    function transform(x, ub, lb, a0)
        return (ub-lb) * (tanh(x/a0)+1) / 2 + lb
    end

    function invTransform(x, ub, lb, a0)
        return a0 * atanh( (lb+ub-2*max.(lb+0.001,min.(ub-0.001,x)) ) / (lb-ub) )
    end

    function dTransform(x, ub, lb, a0)
        return (ub-lb) * sech(x/a0)^2 / (2*a0)
    end

    function fg!(F,G,u)
println("start")
        getF = F != nothing
        getG = G != nothing
        if getG
            G .= 0
        end

        @views state.rho[state.keepInd] .= transform.(u[1,ki], 1, 0, mrbcdi.ra0)
        @views state.ux[state.keepInd] .= transform.(u[2,ki], 1.1*pi, -1.1*pi, mrbcdi.ua0)
        @views state.uy[state.keepInd] .= transform.(u[3,ki], 1.1*pi, -1.1*pi, mrbcdi.ua0)
        @views state.uz[state.keepInd] .= transform.(u[4,ki], 1.1*pi, -1.1*pi, mrbcdi.ua0)
        mrbcdi.losses .= 0
        for i in optimCore
            if state.highStrain && state.rotations == nothing
                @views BcdiCore.setpts!(
                    state.cores[i], 
                    state.xPos[1].+state.ux[state.keepInd], 
                    state.yPos[1].+state.ux[state.keepInd], 
                    state.yPos[1].+state.uz[state.keepInd], 
                    state.rho[state.keepInd], state.ux[state.keepInd], 
                    state.uy[state.keepInd], state.uz[state.keepInd], getG
                )
            elseif state.highStrain
                @views BcdiCore.setpts!(
                    state.cores[i], 
                    state.xPos[i].+state.ux[state.keepInd], 
                    state.yPos[i].+state.ux[state.keepInd], 
                    state.yPos[i].+state.uz[state.keepInd],  
                    state.rho[state.keepInd], state.ux[state.keepInd], 
                    state.uy[state.keepInd], state.uz[state.keepInd], getG
                )
            elseif state.rotations != nothing
                @views BcdiCore.setpts!(
                    state.cores[i],
                    state.xPos[i], state.yPos[i], state.yPos[i],
                    state.rho[state.keepInd], state.ux[state.keepInd], 
                    state.uy[state.keepInd], state.uz[state.keepInd], getG
                )
            else
                @views BcdiCore.setpts!(
                    state.cores[i], state.rho[state.keepInd], state.ux[state.keepInd], 
                    state.uy[state.keepInd], state.uz[state.keepInd], getG
                )
            end
            loss = BcdiCore.loss(state.cores[i], getG, getF, false)
            if getF
                loss .+= BcdiCore.modifyLoss(state.cores[i], mrbcdi.TVRegs[i])
                loss .+= BcdiCore.modifyLoss(state.cores[i], mrbcdi.BetaRegs[i])
                mrbcdi.losses[i:i] .= loss
            end
            if getG
                BcdiCore.modifyDeriv(state.cores[i], mrbcdi.TVRegs[i])
                BcdiCore.modifyDeriv(state.cores[i], mrbcdi.BetaRegs[i])
                @views state.cores[i].rhoDeriv .*= dTransform.(u[1,ki], 1, 0, mrbcdi.ra0)
                @views state.cores[i].uxDeriv .*= dTransform.(u[2,ki], 1.1*pi, -1.1*pi, mrbcdi.ua0)
                @views state.cores[i].uyDeriv .*= dTransform.(u[3,ki], 1.1*pi, -1.1*pi, mrbcdi.ua0)
                @views state.cores[i].uzDeriv .*= dTransform.(u[4,ki], 1.1*pi, -1.1*pi, mrbcdi.ua0)
            end
        end
        if getG
            for i in optimCore
                G[1,ki] .+= state.cores[i].rhoDeriv
                G[2,ki] .+= state.cores[i].uxDeriv
                G[3,ki] .+= state.cores[i].uyDeriv
                G[4,ki] .+= state.cores[i].uzDeriv
            end
            G ./= length(optimCore)

            @views G[1,ki] .*= state.support[state.keepInd]
            @views G[2,ki] .*= state.support[state.keepInd]
            @views G[3,ki] .*= state.support[state.keepInd]
            @views G[4,ki] .*= state.support[state.keepInd]
        end

println(reduce(+, mrbcdi.losses) / length(optimCore))
println(reduce(+, state.rho .> 0.5))
println(maximum(state.rho))
        return reduce(+, mrbcdi.losses) / length(optimCore)
    end

    mrbcdi.allU[1,ki] .= invTransform.(state.rho[state.keepInd], 1, 0, mrbcdi.ra0)
    mrbcdi.allU[2,ki] .= invTransform.(state.ux[state.keepInd], 1.1*pi, -1.1*pi, mrbcdi.ua0)
    mrbcdi.allU[3,ki] .= invTransform.(state.uy[state.keepInd], 1.1*pi, -1.1*pi, mrbcdi.ua0)
    mrbcdi.allU[4,ki] .= invTransform.(state.uz[state.keepInd], 1.1*pi, -1.1*pi, mrbcdi.ua0)

    method = LBFGS(alphaguess=InitialPrevious(alpha=mrbcdi.alpha[]), linesearch=MoreThuente())
    options = Optim.Options(
        iterations=mrbcdi.iterations, g_abstol=-1.0, g_reltol=-1.0,
        x_abstol=-1.0, x_reltol=-1.0, f_abstol=-1.0, f_reltol=-1.0
    )
    objective = Optim.promote_objtype(method, mrbcdi.allU, :finite, true, Optim.only_fg!(fg!))
    init_state = Optim.initial_state(method, options, objective, mrbcdi.allU)
    init_state.alpha = NaN
    res = Optim.optimize(objective, mrbcdi.allU, method, options, init_state)

    mrbcdi.allU .= Optim.minimizer(res)
    mrbcdi.B .= transform.(mrbcdi.allU[2:4,vec(ki)], 1.1*pi, -1.1*pi, mrbcdi.ua0)
    @views mrbcdi.allU[2:4,vec(ki)] .= mrbcdi.B .- 2 .* pi .* (mrbcdi.AInv \ floor.(
        Int64, (mrbcdi.AInv * mrbcdi.B) ./ (2 .* pi) .+ 0.5
    ))

    state.rho[state.keepInd] .= transform.(mrbcdi.allU[1,ki], 1, 0, mrbcdi.ra0)
    state.ux[state.keepInd] .= mrbcdi.allU[2,ki]
    state.uy[state.keepInd] .= mrbcdi.allU[3,ki]
    state.uz[state.keepInd] .= mrbcdi.allU[4,ki]

end

function Base.:*(operator::Operator, state::State)
    operate(operator, state)
    return state
end

function Base.:*(operator1::Operator, operator2::Operator)
    return OperatorList([operator2, operator1])
end

function Base.:^(operator::Operator, pow::Int)
    return OperatorList([operator for i in 1:pow])
end
