# Julia BCDI Documentation

## About

The BYU-CXI research group maintains a suite of Julia packages to solve the Bragg Coherent Diffraction Imaging (BCDI) problem in several different regimes and circumstance. 

- [BcdiCore.jl](https://byu-cxi.github.io/BcdiCore.jl/dev) implements all of the Fourier transforms for the Julia BCDI packages. In addition, BcdiCore calculates the loss function used (either ``L_2`` or the MLE estimator) and derivatives of these loss functions.

- [BcdiTrad.jl](https://byu-cxi.github.io/BcdiTrad.jl/dev) implements projection-based BCDI algorithm. Currently, this is limited to ER, HIO, and shrinkwrap. 

- [BcdiStrain.jl](https://byu-cxi.github.io/BcdiStrain.jl/dev) implements a multi-peak BCDI algorithm developed by the BYU-CXI group. In addition to the alogrithms present in BcdiTrad, BcdiStrain also implements Mount, an operator that switches between peaks.

- [BcdiMeso.jl](https://byu-cxi.github.io/BcdiMeso.jl/dev) implements a BCDI algorithm that solves in the mesoscale regime. Instead of using projections, this algorithm uses a gradient-based optimization scheme. Additionally, BcdiMeso does not assume a small measurement distance away from the peak.

- BcdiAtomic.jl is an upcoming BCDI package that implements a BCDI algorithm that solves at the atomic scale.

- BcdiMulti.jl is an upcoming BCDI package that implements a multiscale BCDI algorithm that solves at both the mesoscale and the atomic scale.

- BcdiSimulate.jl is an upcoming BCDI package that simulates the BCDI problem. Currently, this is only implimented at the atomic scale.
