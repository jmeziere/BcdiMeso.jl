"""
   State(intens, gVecs, recSupport, x, y, z, rho, ux, uy, uz)

Create the reconstruction state. `intens` is a vector of fully measured diffraction
peaks, `gVecs` is a vector of peak locations, and `recSupport` is a vector of masks over
the intens that removes those intenities from the reconstruction process. The
positions of real space points (`x`, `y`, and `z`) must be passed in as well as the
magnitude of the electron density `rho` and the displacement field (`ux`, `uy`, and `uz`).

The initialization process shifts each peak to be centered (i.e. the center of 
mass of the peak is moved to the center of the image). 
"""
struct State{T,I}
    rho::CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}
    ux::CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}
    uy::CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}
    uz::CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}
    disux::CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}
    disuy::CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}
    disuz::CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}
    support::CuArray{Bool, 3, CUDA.Mem.DeviceBuffer}
    cores::Vector{BcdiCore.MesoState}
    shifts::Vector{Tuple{Int64,Int64,Int64}}
    rotations::T
    highStrain::Bool
    xPos::Vector{CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}}
    yPos::Vector{CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}}
    zPos::Vector{CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}}
    keepInd::CuArray{CartesianIndex{3}, I, CUDA.Mem.DeviceBuffer}

    function State(intens, gVecs, recSupport; rotations=nothing, highStrain=false, truncRecSupport=true)
        s = size(intens[1])
        n = rotations == nothing && !highStrain ? size(intens[1]) : reduce(*, size(intens[1]))
        rho = CUDA.zeros(Float64, s)
        ux = CUDA.zeros(Float64, s)
        uy = CUDA.zeros(Float64, s)
        uz = CUDA.zeros(Float64, s)
        if highStrain
            disux = CUDA.zeros(Float64, s)
            disuy = CUDA.zeros(Float64, s)
            disuz = CUDA.zeros(Float64, s)
        else
            disux = CUDA.zeros(Float64, 0,0,0)
            disuy = CUDA.zeros(Float64, 0,0,0)
            disuz = CUDA.zeros(Float64, 0,0,0)
        end
        support = CUDA.ones(Bool, s)

        cores = BcdiCore.MesoState[]
        shifts = Tuple{Int64,Int64,Int64}[]

        xPos = []
        yPos = []
        zPos = []
        if rotations != nothing || highStrain
            nonuniform = true
            if highStrain
                disjoint = true
            else
                disjoint = false
            end
            baseX = zeros(Float64, s)
            baseY = zeros(Float64, s)
            baseZ = zeros(Float64, s)
            for i in 1:s[1]
                for j in 1:s[2]
                    for k in 1:s[3]
                        baseX[i,j,k] = 2*pi*(i-1)/s[1]-pi
                        baseY[i,j,k] = 2*pi*(j-1)/s[2]-pi
                        baseZ[i,j,k] = 2*pi*(k-1)/s[3]-pi
                    end
                end
            end
            baseCuX = CuArray{Float64}(baseX)
            baseCuY = CuArray{Float64}(baseY)
            baseCuZ = CuArray{Float64}(baseZ)
            baseX = vec(baseX)
            baseY = vec(baseY)
            baseZ = vec(baseZ)

            cpuXPos = []
            cpuYPos = []
            cpuZPos = []
            if rotations != nothing
                deleteInd = []
                for i in 1:length(rotations)
                    rot = transpose(rotations[i])
                    x = rot[1,1] .* baseX .+ rot[1,2] .* baseY .+ rot[1,3] .* baseZ
                    y = rot[2,1] .* baseX .+ rot[2,2] .* baseY .+ rot[2,3] .* baseZ
                    z = rot[3,1] .* baseX .+ rot[3,2] .* baseY .+ rot[3,3] .* baseZ

                    push!(cpuXPos, x)
                    push!(cpuYPos, y)
                    push!(cpuZPos, z)
                    append!(deleteInd, findall(x .> pi .|| x .< -pi .|| y .> pi .|| y .< -pi .|| z .> pi .|| z .< -pi))
                end
                deleteInd = sort(unique(deleteInd))
                keepInd = deleteat!(collect(vec(CartesianIndices(s))),deleteInd)
                for i in 1:length(rotations)
                    deleteat!(cpuXPos[i], deleteInd)
                    deleteat!(cpuYPos[i], deleteInd)
                    deleteat!(cpuZPos[i], deleteInd)
                    push!(xPos, CuArray{Float64}(cpuXPos[i]))
                    push!(yPos, CuArray{Float64}(cpuYPos[i]))
                    push!(zPos, CuArray{Float64}(cpuZPos[i]))
                end
            else
                baseX = CuArray{Float64}(baseX)
                baseY = CuArray{Float64}(baseY)
                baseZ = CuArray{Float64}(baseZ)
                push!(xPos, baseX)
                push!(yPos, baseY)
                push!(zPos, baseZ)
                keepInd = collect(vec(CartesianIndices(s)))
            end
            for i in 1:length(intens)
                currIntens, currRecSupport, shift = BcdiCore.centerPeak(intens[i], recSupport[i], "center", truncRecSupport)
                push!(cores, BcdiCore.MesoState("L2", true, currIntens, gVecs[i], currRecSupport, nonuniform, highStrain, disjoint))
                push!(shifts, shift)
            end

            if highStrain
                @views BcdiCore.setpts!(
                    cores[1],
                    xPos[1], yPos[1], zPos[1],
                    rho[keepInd], ux[keepInd], uy[keepInd], uz[keepInd],
                    disux[keepInd], disuy[keepInd], disuz[keepInd], false
                )
            else
                @views BcdiCore.setpts!(
                    cores[1],
                    xPos[1], yPos[1], zPos[1],
                    rho[keepInd], ux[keepInd], uy[keepInd], uz[keepInd], false
                )
            end
            randAngle = CuArray{Float64}(2 .* pi .* rand(size(cores[1].intens)...))
            cores[1].plan \ (sqrt.(cores[1].intens .* exp.(1im .* randAngle)) .* cores[1].recSupport)
#            rho[keepInd] .= abs.(cores[1].plan.realSpace)
#            rho ./= maximum(rho)
            rho .= 0
            CUDA.@allowscalar rho[51,51,51] = 1
        else
            for i in 1:length(intens)
                currIntens, currRecSupport, shift = BcdiCore.centerPeak(intens[i], recSupport[i], "corner", truncRecSupport)
                push!(cores, BcdiCore.MesoState("L2", true, currIntens, gVecs[i], currRecSupport))
                push!(shifts, shift)
            end
            keepInd = vec(collect(CartesianIndices(s)))

            randAngle = CuArray{Float64}(2 .* pi .* rand(size(cores[1].intens)...))
            cores[1].plan \ (sqrt.(cores[1].intens .* exp.(1im .* randAngle)) .* cores[1].recSupport)
            rho .= abs.(cores[1].plan.realSpace)
            rho ./= maximum(rho)

            e1 = div(s[1], 2)
            e2 = div(s[2], 2)
            e3 = div(s[3], 2)
            rho .*= support
        end
        new{typeof(rotations),ndims(keepInd)}(rho,ux,uy,uz,disux,disuy,disuz,support,cores,shifts,rotations,highStrain,xPos,yPos,zPos,keepInd)
    end
end

function StrainToMeso(strainState)
    inSupp = Array(findall(CUDA.CUFFT.fftshift(strainState.traditionals[1].support)))
    s = size(strainState.traditionals[1].support)
    x = zeros(length(inSupp))
    y = zeros(length(inSupp))
    z = zeros(length(inSupp))
    for i in 1:length(inSupp)
        x[i] = 2*pi*(inSupp[i][1]-1)/s[1]
        y[i] = 2*pi*(inSupp[i][2]-1)/s[2]
        z[i] = 2*pi*(inSupp[i][3]-1)/s[3]
    end

    return State(
        [Array(CUDA.CUFFT.fftshift(strainState.traditionals[i].core.intens)) for i in 1:length(strainState.traditionals)], 
        strainState.gVecs,
        [Array(CUDA.CUFFT.fftshift(strainState.traditionals[i].core.recSupport)) for i in 1:length(strainState.traditionals)],
        x, y, z, 
        CUDA.CUFFT.fftshift(strainState.rho)[inSupp], 
        CUDA.CUFFT.fftshift(strainState.ux)[inSupp], 
        CUDA.CUFFT.fftshift(strainState.uy)[inSupp], 
        CUDA.CUFFT.fftshift(strainState.uz)[inSupp]
    )
end
