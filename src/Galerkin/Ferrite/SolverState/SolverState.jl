using Ferrite
export FerriteSolverState

mutable struct FerriteSolverState <: AbstractGalerkinSolver
    ∂Ω # Definition of the boundary
    σ::AbstractVector
    L::AbstractMatrix #Current guess of boundary operator for
    L_fac # Factorized version of K
    Σ::AbstractVector # Singular values of K
    d # metric we use
    ∂d # derivative of the metric
    down # Projection from force vector to coefficients of the basis functions of boundary
    up # Projection from coefficients of the basis functions of boundary to force vector
    R # Some Function that holds the regularizer, a convex lower-semicontinuous function
    r_args # Arguments for the regularizer function
    β::Float64 # Regularization parameter
    τ::Float64 # Learn rate
    num_pairs::Int64 # Number of voltage-current pairs
end

function FerriteSolverState(∂Ω,fe::FerriteFESpace,σ::AbstractVector)

    FerriteSolverState(∂Ω, σ, L, L_fac, Σ, d, ∂d, down, up, R, r_args, β, τ, num_pairs)
end
