"""
   State(intensities, gVecs, recSupport, x, y, z, rho, ux, uy, uz)

Create the reconstruction state. `intensities` is a vector of fully measured diffraction
peaks, `gVecs` is a vector of peak locations, and `recSupport` is a vector of masks over
the intensities that removes those intenities from the reconstruction process. The
positions of real space points (`x`, `y`, and `z`) must be passed in as well as the
magnitude of the electron density `rho` and the displacement field (`ux`, `uy`, and `uz`).

The initialization process shifts each peak to be centered (i.e. the center of 
mass of the peak is moved to the center of the image). 
"""
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

    function State(intensities, gVecs, recSupport, x, y, z, rho, ux, uy, uz)
        s = size(intensities[1])
        h,k,l = BcdiCore.generateRecSpace(s)

        cores = BcdiCore.MesoState[]
        shifts = Tuple{Int64,Int64,Int64}[]
        for i in 1:length(intensities)
            currIntens, currRecSupport, shift = BcdiCore.centerPeak(intensities[i], recSupport[i], "center")
            push!(cores, BcdiCore.MesoState("L2", true, currIntens, gVecs[i], h, k, l, currRecSupport))
            push!(shifts, shift)
        end

        new(x,y,z,rho,ux,uy,uz,cores,shifts)
    end
end
