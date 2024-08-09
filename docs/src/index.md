# BcdiMeso.jl Documentation

## About

Bragg Coherent Diffraction Imaging (BCDI) Meso (Mesoscale) implements phase retrieval for mesoscale models with stochastic gradient descent. Some of the core functionality of this project is implemented in BcdiCore.jl. Additionally, this package is part of a collection of BCDI packages consisting of BcdiAtomic.jl (an atomic scale BCDI solver), BcdiMeso.jl (a mesoscale BCDI Solver), and BcdiMulti.jl (a multiscale BCDI solver).

While this package is marked as BCDI specific, the methods are more general and can be used in many phase retrieval problems. In the future, this package may be incorporated into a more general phase retrieval package.

Currently, this entire package must be run with access to GPUs. This may change in the future (especially if Issues requesting it are opened), but for our research group, using GPUs is a necessity.

## Installation

Currently, BcdiStrain.jl is registered in the Julia general registry and can be installed by running in the REPL package manager (```]```):

```
add BcdiMeso
```
