"""
# AbstractGelerkin
This modul is supposed to give a scaffold for numerical solvers based on Galerkin methods like FEM, Spectral methods (like Chebyshev, FFT, Polynomials) or Kernel methods.
Is goes that it implements:
- A hilbertspace that allows for one or multiple forms of scalar product, especially H¹ and L².
- a general adjoint-state-method that drives the iterative upgrade by calculating a gradient
- some sort of adaptive method that refines the parameters of the interpolation space (like adaptive meshing)
- some sort of boundary projection and back projection that manages parametrization of the boundary
- Some sort of Metric or Pseudometric that measures the error.
- Some sort of Regularization.
- Some sort of

"""
#module AbstractGalerkin

export AbstractHilbertSpace, AbstractGalerkinSolver, AbstractAdjointSolver, AbstractBoundaryPair, AbstractProblem


abstract type AbstractHilbertSpace end # This is supposed to hold all information about the space
abstract type AbstractGalerkinSolver end # This is supposed to hold all the information that the solver needs
abstract type AbstractAdjointSolver end # This is supposed to be all needed to get a gradient δσ for the update of σ
abstract type AbstractBoundaryPair end # This is supposed to halod all the information for a voltage-current boundary pair.

abstract type AbstractProblem end # This is supposed to hold all the information that the problem needs

abstract type GalerkinOptState end

#end # Module
