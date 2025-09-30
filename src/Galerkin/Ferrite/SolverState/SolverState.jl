using Ferrite
export FerriteSolverState

mutable struct FerriteSolverState <: AbstractGalerkinSolver
    σ::AbstractVector
    L::AbstractMatrix
    L_fac # Factorized version of K
    Σ::AbstractVector # Singular values of K
    d # metric we use
    ∂d # derivative of the metric

end
