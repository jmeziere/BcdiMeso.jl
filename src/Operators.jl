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
        kiArr[vec(state.keepInd)] .= 1:length(state.keepInd)
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
            lambdaTV * reduce(+, state.cores[i].intens) / reduce(+, state.support), neighbors
        ) for i in 1:length(state.cores)]
        BetaRegs = [BcdiCore.BetaReg(
            lambdaBeta * reduce(+, state.cores[i].intens) / reduce(+, state.support), a, b, c
        ) for i in 1:length(state.cores)]
        new{ndims(allU),typeof(TVRegs),typeof(BetaRegs)}(
            numPeaks, primitiveRecipLattice, allU, B, losses,
            iterations, 1, 5000, alpha, TVRegs, BetaRegs
        )
    end
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
                state.uy[state.keepInd], state.uz[state.keepInd], getG
            )
        elseif state.rotations == nothing && state.highStrain
            @views BcdiCore.setpts!(
                state.cores[i],
                state.xPos[1],
                state.yPos[1],
                state.zPos[1],
                state.rho[state.keepInd], state.ux[state.keepInd],
                state.uy[state.keepInd], state.uz[state.keepInd], getG
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
println(maximum(abs.(state.cores[i].rhoDeriv)))
            BcdiCore.modifyDeriv(state.cores[i], mrbcdi.TVRegs[i])
            BcdiCore.modifyDeriv(state.cores[i], mrbcdi.BetaRegs[i])
println(maximum(abs.(state.cores[i].rhoDeriv)))
            @views state.cores[i].rhoDeriv[ki] .*= dTransform.(u[1,ki], 1, 0, mrbcdi.ra0)
            @views state.cores[i].uxDeriv[ki] .*= dTransform.(u[2,ki], 1.1*pi, -1.1*pi, mrbcdi.ua0)
            @views state.cores[i].uyDeriv[ki] .*= dTransform.(u[3,ki], 1.1*pi, -1.1*pi, mrbcdi.ua0)
            @views state.cores[i].uzDeriv[ki] .*= dTransform.(u[4,ki], 1.1*pi, -1.1*pi, mrbcdi.ua0)
        end
    end
    if getG
        for i in optimCore
            G[1,ki] .+= state.cores[i].rhoDeriv[ki]
            G[2,ki] .+= state.cores[i].uxDeriv[ki]
            G[3,ki] .+= state.cores[i].uyDeriv[ki]
            G[4,ki] .+= state.cores[i].uzDeriv[ki]
        end
        G ./= length(optimCore)
    end

println(reduce(+, state.rho .> 0.5))
println(reduce(+, state.support))
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
end

struct Center <: Operator
    xArr::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    yArr::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    zArr::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    space::CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}

    function Center(state)
        s = size(state.rho)
        xArr = zeros(Int64, s)
        yArr = zeros(Int64, s)
        zArr = zeros(Int64, s)
        for i in 1:s[1]
            for j in 1:s[2]
                for k in 1:s[3]
                    xArr[i,j,k] = i
                    yArr[i,j,k] = j
                    zArr[i,j,k] = k
                end
            end
        end

        space = CUDA.zeros(Float64, s)
        support = CUDA.zeros(Int64, s)

        new(xArr, yArr, zArr, space)
    end
end

function operate(center::Center, state::State)
    s = size(center.space)
    if !state.highStrain && state.rotations == nothing
        circshift!(center.space, state.rho, [s[1]//2,s[2]//2,s[3]//2])
    else
        center.space .= state.rho
    end

    n = reduce(+, center.space)
    cenX = round(Int32, mapreduce((r,x)->r*x, +, center.space, center.xArr)/n)
    cenY = round(Int32, mapreduce((r,x)->r*x, +, center.space, center.yArr)/n)
    cenZ = round(Int32, mapreduce((r,x)->r*x, +, center.space, center.zArr)/n)

    circshift!(center.space, state.rho, [s[1]//2+1-cenX, s[2]//2+1-cenY, s[3]//2+1-cenZ])
    state.rho .= center.space
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
