"""
    ModularEIT

Modular building blocks for Electrical Impedance Tomography (EIT): meshes, electrode
models, forward solvers, regularizers and reconstruction algorithms that can be
combined freely.

!!! warning "Mock implementation"
    The current code base is a placeholder used to set up the documentation
    pipeline. The numerics are deliberately simplistic.
"""
module ModularEIT

using LinearAlgebra

export EITMesh, circle_mesh, nnodes, nelements
export Electrode, ring_electrodes
export ForwardProblem, solve_forward, jacobian
export Regularizer, Tikhonov, TotalVariation, penalty, gradient
export ReconstructionResult, reconstruct

include("mesh.jl")
include("electrodes.jl")
include("forward.jl")
include("regularization.jl")
include("reconstruction.jl")

end # module ModularEIT
