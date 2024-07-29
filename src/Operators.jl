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

struct LinearStep
    x::CUDA.CuArray{Float64, 1, CUDA.DeviceMemory}
    y::CUDA.CuArray{Float64, 1, CUDA.DeviceMemory}
    z::CUDA.CuArray{Float64, 1, CUDA.DeviceMemory}
    alpha0::Float64
    dphi_0_previous::Base.RefValue{Float64}

    function LinearStep(state, alpha0)
        new(state.x, state.y, state.z, alpha0, Ref{Float64}(NaN))
    end
end

function (is::LinearStep)(ls, state, phi_0, dphi_0, df)
    @views minX = mapreduce(
        (up,u,x)-> up >= 0 ? (2*pi-x-u)/up : -(x+u)/up, min, 
        state.s[length(is.x)+1:2*length(is.x)], state.x[length(is.x)+1:2*length(is.x)], is.x
    )
    @views minY = mapreduce(
        (up,u,x)-> up >= 0 ? (2*pi-x-u)/up : -(x+u)/up, min, 
        state.s[2*length(is.x)+1:3*length(is.x)], 
        state.x[2*length(is.x)+1:3*length(is.x)], is.y
    )
    @views minZ = mapreduce(
        (up,u,x)-> up >= 0 ? (2*pi-x-u)/up : -(x+u)/up, min,
        state.s[3*length(is.x)+1:end], 
        state.x[3*length(is.x)+1:end], is.z
    )
    max_alpha = min(minX, minY, minZ)
    if !isfinite(is.dphi_0_previous[]) || !isfinite(state.alpha)
        # If we're at the first iteration
        alphaguess = is.alpha0
    else
        # state.alpha is the previously used step length
        alphaguess = state.alpha * is.dphi_0_previous[] / dphi_0
    end
    is.dphi_0_previous[] = dphi_0
    state.alpha = min(alphaguess, max_alpha)
end

struct OptimizeState <: Operator
    numPeaks::Int64
    AInv::CuArray{Float64, 2, CUDA.Mem.DeviceBuffer}
    allU::CuArray{Float64, 2, CUDA.Mem.DeviceBuffer}
    B::CuArray{Float64, 2, CUDA.Mem.DeviceBuffer}

    function OptimizeState(state, primitiveRecipLattice, numPeaks)
        allU = CUDA.zeros(3, length(state.ux))
        B = CUDA.zeros(3, length(state.ux))
        new(numPeaks, primitiveRecipLattice, allU, B)
    end
end

function operate(optimizeState::OptimizeState, state)
    optimCore = rand(1:length(state.cores), optimizeState.numPeaks)

    function fg!(F,G,u)
        getF = F != nothing
        getG = G != nothing
        if getG
            G .= 0
        end

        @views rho = u[1:length(state.ux)]
        @views ux = u[length(state.ux)+1:2*length(state.ux)]
        @views uy = u[2*length(state.ux)+1:3*length(state.ux)]
        @views uz = u[3*length(state.ux)+1:end]

        for i in optimCore
            BcdiCore.setpts!(state.cores[i], state.x, state.y, state.z, rho, ux, uy, uz, getG)
            loss = BcdiCore.loss(state.cores[i], getG, getF, false)
            if getF
                F += loss
            end
            if getG
                G[1:length(state.ux)] .+= state.cores[i].rhoDeriv
                G[length(state.ux)+1:2*length(state.ux)] .+= state.cores[i].uxDeriv
                G[2*length(state.ux)+1:3*length(state.ux)] .+= state.cores[i].uyDeriv
                G[3*length(state.ux)+1:end] .+= state.cores[i].uzDeriv
            end
        end
        return F
    end

    res = Optim.minimizer(Optim.optimize(
        Optim.only_fg!(fg!), vcat(state.rho, state.ux, state.uy, state.uz),
        LBFGS(alphaguess=LinearStep(state, 1e-12), linesearch=MoreThuente()),
        Optim.Options(iterations=1, g_abstol=-1.0, g_reltol=-1.0, x_abstol=-1.0, x_reltol=-1.0, f_abstol=-1.0, f_reltol=1e-3)
    ))

    state.rho .= res[1:length(state.ux)]
    state.ux .= res[length(state.ux)+1:2*length(state.ux)]
    state.uy .= res[2*length(state.ux)+1:3*length(state.ux)]
    state.uz .= res[3*length(state.ux)+1:end]

    optimizeState.allU[1,:] .= state.ux
    optimizeState.allU[2,:] .= state.uy
    optimizeState.allU[3,:] .= state.uz

    optimizeState.B .= floor.(Int64, (optimizeState.AInv * optimizeState.allU) ./ (2 .* pi) .+ 0.5)
    optimizeState.allU .-= 2 .* pi .* (optimizeState.AInv \ optimizeState.B)
    state.ux .= optimizeState.allU[1,:]
    state.uy .= optimizeState.allU[2,:]
    state.uz .= optimizeState.allU[3,:]
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
