struct State
    x::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
    y::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
    z::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
    rho::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
    ux::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
    uy::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
    uz::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
    cores::Vector{BcdiCore.MesoState}
    shifts::Vector{Tuple{Int64,Int64,Int64}}

    function State(intens, gVecs, recSupport, x, y, z, rho, ux, uy, uz)
        s = size(intens[1])
        h,k,l = BcdiCore.generateRecSpace(s)

        cores = BcdiCore.MesoState[]
        shifts = Tuple{Int64,Int64,Int64}[]
        for i in 1:length(intens)
            currIntens, currRecSupport, shift = BcdiCore.centerPeak(intens[i], recSupport[i], "center")
            push!(cores, BcdiCore.MesoState("L2", true, currIntens, gVecs[i], h, k, l, currRecSupport))
            push!(shifts, shift)
        end

        new(x,y,z,rho,ux,uy,uz,cores,shifts)
    end
end
