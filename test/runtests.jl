using BcdiMeso
using Test
using CUDA
using ForwardDiff

function transform(x, ub, lb, a0)
    return (ub-lb) * (tanh(x/a0)+1) / 2 + lb
end


function invTransform(x, ub, lb, a0)
    return a0 * atanh( (lb+ub-2*max.(lb+0.001,min.(ub-0.001,x)) ) / (lb-ub) )
end

function mesoModel(x, y, z, rho, ux, uy, uz, ra0, ua0, lambdaT, lambdaB,a,b,c, h, k, l, G, intens, recSupport, neighbors)
    rho = transform.(rho, 1, 0, ra0)
    ux = transform.(ux, 1.1*pi, -1.1*pi, ua0)
    uy = transform.(uy, 1.1*pi, -1.1*pi, ua0)
    uz = transform.(uz, 1.1*pi, -1.1*pi, ua0)
    diffType = Float64
    if typeof(rho[1]) != Float64
        diffType = typeof(rho[1])
    elseif typeof(ux[1]) != Float64
        diffType = typeof(ux[1])
    elseif typeof(uy[1]) != Float64
        diffType = typeof(uy[1])
    elseif typeof(uz[1]) != Float64
        diffType = typeof(uz[1])
    end
    recipSpace = zeros(Complex{diffType}, 4,4,4)
    for i in 1:length(h)
        for j in 1:length(x)
            recipSpace[i] += rho[j] * exp(-1im * (x[j] * h[i] + y[j] * k[i] + z[j] * l[i] + ux[j] * (G[1]+h[i]) + uy[j] * (G[2]+k[i]) + uz[j] * (G[3]+l[i])))
        end
    end
    absRecipSpace = abs.(recipSpace) .* recSupport
    sqIntens = sqrt.(intens) .* recSupport
    c = mapreduce((sqi,absr)-> sqi*absr, +, sqIntens, absRecipSpace)/mapreduce(x -> x^2, +, absRecipSpace)
    loss = mapreduce((sqi,absr) -> (c*absr - sqi)^2, +, sqIntens, absRecipSpace)/length(intens)

    realSpace = rho .* exp.(-1im .* (ux .* G[1] .+ uy .* G[2] .+ uz .* G[3]))
    for i in 1:length(realSpace)
        for j in 1:6
            loss += abs(realSpace[i] - realSpace[neighbors[j,i]]) * lambdaT/6
        end
        loss += (abs(realSpace[i])+0.001)^a*(1.001-abs(realSpace[i]))^b+c*abs(realSpace[i]) * lambdaB
    end
    return loss
end

function mesoModel(rho, ux, uy, uz, ra0, ua0, lambdaT, lambdaB,a,b,c, G, intens, recSupport, neighbors)
    rho = transform.(rho, 1, 0, ra0)
    ux = transform.(ux, 1.1*pi, -1.1*pi, ua0)
    uy = transform.(uy, 1.1*pi, -1.1*pi, ua0)
    uz = transform.(uz, 1.1*pi, -1.1*pi, ua0)
    diffType = Float64
    if typeof(rho[1]) != Float64
        diffType = typeof(rho[1])
    elseif typeof(ux[1]) != Float64
        diffType = typeof(ux[1])
    elseif typeof(uy[1]) != Float64
        diffType = typeof(uy[1])
    elseif typeof(uz[1]) != Float64
        diffType = typeof(uz[1])
    end
    recipSpace = zeros(Complex{diffType}, 10,10,10)
    s1 = size(intens,1)
    s2 = size(intens,2)
    s3 = size(intens,3)
    for i in 0:s1-1
    for j in 0:s2-1
    for k in 0:s3-1
        for l in 0:s1-1
        for m in 0:s2-1
        for n in 0:s3-1
            recipSpace[i+1,j+1,k+1] += rho[l+1,m+1,n+1] * exp(-1im * (ux[l+1,m+1,n+1]*G[1] + uy[l+1,m+1,n+1]*G[2] + uz[l+1,m+1,n+1]*G[3])) * exp(-1im * 2 * pi * (i*l/s1+j*m/s2+k*n/s3))
        end
        end
        end
    end
    end
    end
    absRecipSpace = abs.(recipSpace) .* recSupport
    sqIntens = sqrt.(intens) .* recSupport
    cost = mapreduce((sqi,absr)-> sqi*absr, +, sqIntens, absRecipSpace)/mapreduce(x -> x^2, +, absRecipSpace)
    loss = mapreduce((sqi,absr) -> (cost*absr - sqi)^2, +, sqIntens, absRecipSpace)/length(intens)

    realSpace = rho .* exp.(-1im .* (ux .* G[1] .+ uy .* G[2] .+ uz .* G[3]))
    for i in 1:length(realSpace)
        for j in 1:6
            loss += abs(realSpace[i] - realSpace[neighbors[j,i]]) * lambdaT/6
        end
    end
    for i in 1:length(realSpace)
        loss += ((abs(realSpace[i])+0.001)^a*(1.001-abs(realSpace[i]))^b+c*abs(realSpace[i])) * lambdaB
    end
    return loss
end

@testset "BcdiMeso.jl" begin
    recPrimLatt  = [1 0 0;0 1 0;0 0 1]
    intens = [rand(1:100,10,10,10)]
    gVecs = [rand(3)]
    recSupport = [ones(Bool,10,10,10)]
    state = BcdiMeso.State(intens, gVecs, recSupport)
    lambdaT = 1
    lambdaB = 5
    a = 0.25
    b = 1.0
    c = -1.0
    mrbcdi = BcdiMeso.MRBCDI(state, recPrimLatt, 1, 1, lambdaT, lambdaB, a,b,c, 1e-8)

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
    neighbors = Array(neighbors)

    u = CUDA.zeros(Float64, 4,size(state.keepInd)...)
    G = CUDA.zeros(Float64, 4,size(state.keepInd)...)
    ki = CartesianIndices(state.keepInd)
    u[1,ki] .= invTransform.(state.rho[state.keepInd], 1, 0, mrbcdi.ra0)
    u[2,ki] .= invTransform.(state.ux[state.keepInd], 1.1*pi, -1.1*pi, mrbcdi.ua0)
    u[3,ki] .= invTransform.(state.uy[state.keepInd], 1.1*pi, -1.1*pi, mrbcdi.ua0)
    u[4,ki] .= invTransform.(state.uz[state.keepInd], 1.1*pi, -1.1*pi, mrbcdi.ua0)
    testee = BcdiMeso.fgMRBCDI!(0.0,G,u,mrbcdi,state)

    rho = reshape(Array(u[1,ki]),10,10,10)
    ux =  reshape(Array(u[2,ki]),10,10,10)
    uy =  reshape(Array(u[3,ki]),10,10,10)
    uz =  reshape(Array(u[4,ki]),10,10,10)
    recSupport = Array(state.cores[1].recSupport)
    intens = Array(state.cores[1].intens)

    tester = mesoModel(
        rho, ux, uy, uz, mrbcdi.ra0, mrbcdi.ua0, 
        lambdaT * sqrt(reduce(+, intens)) / length(intens), 
        lambdaB * sqrt(reduce(+, intens)) / length(intens),
        a,b,c, gVecs[1], intens, recSupport, neighbors
    )
    rhoDeriv = ForwardDiff.gradient(rhop -> mesoModel(
        rhop, ux, uy, uz, mrbcdi.ra0, mrbcdi.ua0,
        lambdaT * sqrt(reduce(+, intens)) / length(intens),
        lambdaB * sqrt(reduce(+, intens)) / length(intens),
        a,b,c, gVecs[1], intens, recSupport, neighbors
    ), rho)
    uxDeriv = ForwardDiff.gradient(uxp -> mesoModel(
        rho, uxp, uy, uz, mrbcdi.ra0, mrbcdi.ua0,
        lambdaT * sqrt(reduce(+, intens)) / length(intens),
        lambdaB * sqrt(reduce(+, intens)) / length(intens),
        a,b,c, gVecs[1], intens, recSupport, neighbors
    ), ux)
    uyDeriv = ForwardDiff.gradient(uyp -> mesoModel(
        rho, ux, uyp, uz, mrbcdi.ra0, mrbcdi.ua0,
        lambdaT * sqrt(reduce(+, intens)) / length(intens),
        lambdaB * sqrt(reduce(+, intens)) / length(intens),
        a,b,c, gVecs[1], intens, recSupport, neighbors
    ), uy)
    uzDeriv = ForwardDiff.gradient(uzp -> mesoModel(
        rho, ux, uy, uzp, mrbcdi.ra0, mrbcdi.ua0,
        lambdaT * sqrt(reduce(+, intens)) / length(intens),
        lambdaB * sqrt(reduce(+, intens)) / length(intens),
        a,b,c, gVecs[1], intens, recSupport, neighbors
    ), uz)

    @test @CUDA.allowscalar isapprox(testee[1], tester, atol=1e-6)
    @test all(isapprox.(Array(G[1,ki]), rhoDeriv[ki], atol=1e-6))
    @test all(isapprox.(Array(G[2,ki]), uxDeriv[ki], atol=1e-6))
    @test all(isapprox.(Array(G[3,ki]), uyDeriv[ki], atol=1e-6))
    @test all(isapprox.(Array(G[4,ki]), uzDeriv[ki], atol=1e-6))
end
