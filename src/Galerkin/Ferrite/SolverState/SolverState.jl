using Ferrite
using Enzyme


export FerriteSolverState

mutable struct FerriteSolverState <: AbstractGalerkinSolver
    ∂Ω # Definition of the boundary
    σ::AbstractVector # Conductivity values
    δ::AbstractVector # Update of the conductivity
    L::AbstractMatrix #Current guess of boundary operator for Neumann boundary
    L_fac # Factorized version of K
    LD::Union{AbstractMatrix,Nothing} # Current guess of boundary operator for dirichlet boundary
    LD_fac
    Σ::AbstractVector # Singular values of K
    d # metric we use: d: (x, y) -> norm(x - y)^2 by default
    ∂d # derivative of the metric after x: (x, y) -> 2 * (x - y) by default
    R_diff # Some Function that holds the differentiable part of the regularizer
    R_ndiff # Some Function that holds the non-differentiable part of the regularizer required to be a convex lower-semicontinuous function
    R_diff_args # Arguments for the differentiable regularizer function
    R_ndiff_args # Arguments for the non-differentiable regularizer function
    β_diff::Float64 # Regularization parameter for differentiable part
    β_ndiff::Float64 # Regularization parameter for non-differentiable part
    τ::Float64 # Learn rate
    num_pairs::Int64 # Number of voltage-current pairs
    steps::Int64 # Number of steps taken
end


function FerriteSolverState(fe::FerriteFESpace, σ::AbstractVector)
    d = (x, y) -> norm(x - y)^2
    ∂d = (x, y) -> 2 * (x - y)
    FerriteSolverState(fe, σ, d, ∂d)
end

makegradₓ(d) = (x, y) -> Enzyme.gradient(Reverse, Const(d), x, y)[1]

function FerriteSolverState(fe::FerriteFESpace, σ::AbstractVector, d)
    ∂d = makegradₓ(d)
    FerriteSolverState(fe, σ, d, ∂d)
end

function FerriteSolverState(fe::FerriteFESpace, σ::AbstractVector, d, ∂d)
    ∂Ω = fe.∂Ω
    δ = zeros(fe.n)

    L = assemble_L(fe, σ)
    Σ = zeros(fe.m - 1)

    FerriteSolverState(∂Ω, σ, δ, L, nothing, nothing, nothing, Σ, d, ∂d, nothing, nothing, nothing, nothing, 0.0, 0.0, 0.1, 0, 0)
end
