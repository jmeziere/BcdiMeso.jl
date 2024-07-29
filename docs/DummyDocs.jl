module DummyDocs

export State,OptimizeState

"""
    State(intens, gVecs, recSupport, x, y, z, rho, ux, uy, uz)

Create the reconstruction state.
"""
function State(intens, gVecs, recSupport, x, y, z, rho, ux, uy, uz)
end

"""
    OptimizeState(state, primitiveRecipLattice, numPeaks)

Create an object that performs an iteration of stochastic gradient descent.
"""
function OptimizeState(state, primitiveRecipLattice, numPeaks)
end

end
