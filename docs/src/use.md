# Overview

BcdiMeso implements phase retrieval in operator-style format. This means that the multiplication (*) and power (^) operators are used to apply operators to some current state. This may look like the following:

```
state = State(intens, gVecs, recSupport, x, y, z, rho, ux, uy, uz)
optimizeState = OptimizeState(state, primitiveRecipLattice, numPeaks)

optimizeState^100 * state
```

This short script applies 100 stochastic gradient descent iterations iterations. This makes it easy to implement very complex recipes for phase retrieval algorithms.

# API

```@docs
State(intens, gVecs, recSupport, x, y, z, rho, ux, uy, uz)
OptimizeState(state, primitiveRecipLattice, numPeaks)
```
