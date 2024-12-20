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

        function findNeigh(neighInd, kiInd, neighbors)
            if kiInd == 0
                return
            end
            neighbors[kiInd] = neighInd
        end

        kiArr = CUDA.zeros(Int64, size(state.cores[1].intens))
        kiArr[state.keepInd] .= 1:length(state.keepInd)
        for i in 1:length(inds)
            neighArr = circshift(kiArr, Tuple(inds[i]))
            findNeigh.(neighArr, kiArr, Ref(view(neighbors,i,:)))
        end

        if state.highStrain
            allU = CUDA.zeros(Float64, 7, size(state.keepInd)...)
        else
            allU = CUDA.zeros(Float64, 4, size(state.keepInd)...)
        end
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
            iterations, 1, 5000, alpha, TVRegs, BetaRegs
        )
    end
end

struct TVReg
    lambda::Float64
    neighbors::CuArray{Int64, 2, CUDA.Mem.DeviceBuffer}

    function TVReg(lambda, neighbors)
        newNeighs = CUDA.zeros(Int64, size(neighbors))
        newNeighs[neighbors .!= nothing] .= neighbors[neighbors .!= nothing]
        new(lambda, newNeighs)
    end
end

function modifyDeriv(rho, ux, uy, uz, state, reg::TVReg)
        for i in 1:6
            @views inds = reg.neighbors[i,:] .!= 0
            @views neighs = reg.neighbors[i,inds]

            @views state.rhoDeriv[inds] .+= reg.lambda .* sign.(rho[inds] .- rho[reg.neighbors[i,inds]]) ./ 3
            @views state.uxDeriv[inds] .+= reg.lambda .* sign.(ux[inds] .- ux[reg.neighbors[i,inds]]) ./ 3
            @views state.uyDeriv[inds] .+= reg.lambda .* sign.(uy[inds] .- uy[reg.neighbors[i,inds]]) ./ 3
            @views state.uzDeriv[inds] .+= reg.lambda .* sign.(uz[inds] .- uz[reg.neighbors[i,inds]]) ./ 3
        end
end

function modifyLoss(rho, ux, uy, uz, reg::TVReg)
    mLoss = CUDA.zeros(Float64, 1)
    for i in 1:6
        @views inds = reg.neighbors[i,:] .!= 0
        @views mLoss .+= mapreduce(
            (r,n) -> abs(r - n), +,
            rho[inds], rho[reg.neighbors[i,inds]], dims=(1)
        )
        @views mLoss .+= mapreduce(
            (u,n) -> abs(u - n), +,
            ux[inds], ux[reg.neighbors[i,inds]], dims=(1)
        )
        @views mLoss .+= mapreduce(
            (u,n) -> abs(u - n), +,
            uy[inds], uy[reg.neighbors[i,inds]], dims=(1)
        )
        @views mLoss .+= mapreduce(
            (u,n) -> abs(u - n), +,
            uz[inds], uz[reg.neighbors[i,inds]], dims=(1)
        )
    end
    return reg.lambda .* mLoss ./ 6
end

function transform(x, ub, lb, a0)
    return (ub-lb) * (tanh(x/a0)+1) / 2 + lb
end

function invTransform(x, ub, lb, a0)
    return a0 * atanh( (lb+ub-2*max.(lb+0.001,min.(ub-0.001,x)) ) / (lb-ub) )
end

function dTransform(x, ub, lb, a0)
    return (ub-lb) * sech(x/a0)^2 / (2*a0)
end

function fgMRBCDI!(F,G,u,mrbcdi,state,optimCore)
    getF = F != nothing
    getG = G != nothing
    if getG
        G .= 0
    end

    ki = CartesianIndices(state.keepInd)
    @views state.rho[state.keepInd] .= transform.(u[1,ki], 1, 0, mrbcdi.ra0)
    @views state.ux[state.keepInd] .= transform.(u[2,ki], 1.1*pi, -1.1*pi, mrbcdi.ua0)
    @views state.uy[state.keepInd] .= transform.(u[3,ki], 1.1*pi, -1.1*pi, mrbcdi.ua0)
    @views state.uz[state.keepInd] .= transform.(u[4,ki], 1.1*pi, -1.1*pi, mrbcdi.ua0)
    if state.highStrain
        @views state.disux[state.keepInd] .= transform.(u[5,ki], 1.1*pi, -1.1*pi, mrbcdi.ua0)
        @views state.disuy[state.keepInd] .= transform.(u[6,ki], 1.1*pi, -1.1*pi, mrbcdi.ua0)
        @views state.disuz[state.keepInd] .= transform.(u[7,ki], 1.1*pi, -1.1*pi, mrbcdi.ua0)
    end
    mrbcdi.losses .= 0
println("start")
    for i in optimCore
        if state.rotations != nothing && !state.highStrain
            @views BcdiCore.setpts!(
                state.cores[i],
                state.xPos[i], state.yPos[i], state.zPos[i],
                state.rho[state.keepInd], state.ux[state.keepInd],
                state.uy[state.keepInd], state.uz[state.keepInd], getG
            )
        elseif state.rotations != nothing && state.highStrain
            @views BcdiCore.setpts!(
                state.cores[i],
                state.xPos[i],
                state.yPos[i],
                state.zPos[i],
                state.rho[state.keepInd], state.ux[state.keepInd],
                state.uy[state.keepInd], state.uz[state.keepInd],
                state.disux[state.keepInd], state.disuy[state.keepInd], 
                state.disuz[state.keepInd], getG
            )
        elseif state.rotations == nothing && state.highStrain
            @views BcdiCore.setpts!(
                state.cores[i],
                state.xPos[1],
                state.yPos[1],
                state.zPos[1],
                state.rho[state.keepInd], state.ux[state.keepInd],
                state.uy[state.keepInd], state.uz[state.keepInd],
                state.disux[state.keepInd], state.disuy[state.keepInd], 
                state.disuz[state.keepInd], getG
            )
        elseif state.rotations == nothing && !state.highStrain
            @views BcdiCore.setpts!(
                state.cores[i], state.rho, state.ux,
                state.uy, state.uz, getG
            )
        end
        loss = BcdiCore.loss(state.cores[i], getG, getF, false)
        if getF
println(loss)
            loss .+= BcdiCore.modifyLoss(state.cores[i], mrbcdi.TVRegs[i])
            loss .+= BcdiCore.modifyLoss(state.cores[i], mrbcdi.BetaRegs[i])
println(loss)
            mrbcdi.losses[i:i] .= loss
        end
        if getG
            BcdiCore.modifyDeriv(state.cores[i], mrbcdi.TVRegs[i])
            BcdiCore.modifyDeriv(state.cores[i], mrbcdi.BetaRegs[i])
            @views state.cores[i].rhoDeriv[ki] .*= dTransform.(u[1,ki], 1, 0, mrbcdi.ra0)
            @views state.cores[i].uxDeriv[ki] .*= dTransform.(u[2,ki], 1.1*pi, -1.1*pi, mrbcdi.ua0)
            @views state.cores[i].uyDeriv[ki] .*= dTransform.(u[3,ki], 1.1*pi, -1.1*pi, mrbcdi.ua0)
            @views state.cores[i].uzDeriv[ki] .*= dTransform.(u[4,ki], 1.1*pi, -1.1*pi, mrbcdi.ua0)
            if state.highStrain
                @views state.cores[i].disuxDeriv[ki] .*= dTransform.(u[5,ki], 1.1*pi, -1.1*pi, mrbcdi.ua0)
                @views state.cores[i].disuyDeriv[ki] .*= dTransform.(u[6,ki], 1.1*pi, -1.1*pi, mrbcdi.ua0)
                @views state.cores[i].disuzDeriv[ki] .*= dTransform.(u[7,ki], 1.1*pi, -1.1*pi, mrbcdi.ua0)
            end
        end
    end
    if getG
        for i in optimCore
            G[1,ki] .+= state.cores[i].rhoDeriv[ki]
            G[2,ki] .+= state.cores[i].uxDeriv[ki]
            G[3,ki] .+= state.cores[i].uyDeriv[ki]
            G[4,ki] .+= state.cores[i].uzDeriv[ki]
            if state.highStrain
                G[5,ki] .+= state.cores[i].disuxDeriv[ki]
                G[6,ki] .+= state.cores[i].disuyDeriv[ki]
                G[7,ki] .+= state.cores[i].disuzDeriv[ki]
            end
        end
        G ./= length(optimCore)
    end

println(reduce(+, state.rho .> 0.5))
    return reduce(+, mrbcdi.losses) / length(optimCore)
end

function operate(mrbcdi::MRBCDI, state)
    ki = CartesianIndices(state.keepInd)
    optimCore = StatsBase.sample(1:length(state.cores), mrbcdi.numPeaks, replace=false)
    iters = 0

    mrbcdi.allU[1,ki] .= invTransform.(state.rho[state.keepInd], 1, 0, mrbcdi.ra0)
    mrbcdi.allU[2,ki] .= invTransform.(state.ux[state.keepInd], 1.1*pi, -1.1*pi, mrbcdi.ua0)
    mrbcdi.allU[3,ki] .= invTransform.(state.uy[state.keepInd], 1.1*pi, -1.1*pi, mrbcdi.ua0)
    mrbcdi.allU[4,ki] .= invTransform.(state.uz[state.keepInd], 1.1*pi, -1.1*pi, mrbcdi.ua0)
    if state.highStrain
        mrbcdi.allU[5,ki] .= invTransform.(state.disux[state.keepInd], 1.1*pi, -1.1*pi, mrbcdi.ua0)
        mrbcdi.allU[6,ki] .= invTransform.(state.disuy[state.keepInd], 1.1*pi, -1.1*pi, mrbcdi.ua0)
        mrbcdi.allU[7,ki] .= invTransform.(state.disuz[state.keepInd], 1.1*pi, -1.1*pi, mrbcdi.ua0)
    end

    method = LBFGS(alphaguess=InitialPrevious(alpha=mrbcdi.alpha[]), linesearch=MoreThuente())
    options = Optim.Options(
        iterations=mrbcdi.iterations, g_abstol=-1.0, g_reltol=-1.0,
        x_abstol=-1.0, x_reltol=-1.0, f_abstol=-1.0, f_reltol=-1.0
    )
    objective = Optim.promote_objtype(
        method, mrbcdi.allU, :finite, true, 
        Optim.only_fg!((F,G,u)->fgMRBCDI!(F,G,u,mrbcdi,state,optimCore))
    )
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
    if state.highStrain
        state.disux[state.keepInd] .= transform.(mrbcdi.allU[5,ki], 1.1*pi, -1.1*pi, mrbcdi.ua0)
        state.disuy[state.keepInd] .= transform.(mrbcdi.allU[6,ki], 1.1*pi, -1.1*pi, mrbcdi.ua0)
        state.disuz[state.keepInd] .= transform.(mrbcdi.allU[7,ki], 1.1*pi, -1.1*pi, mrbcdi.ua0)
    end
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
