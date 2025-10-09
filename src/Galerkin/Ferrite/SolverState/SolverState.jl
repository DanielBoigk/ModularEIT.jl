using Ferrite
using Enzyme


export FerriteSolverState


"""
    mutable struct FerriteSolverState <: AbstractGalerkinSolver

Encapsulates the state of an EIT solver using the Ferrite FEM framework.
Holds the conductivity, PDE operators, regularizers, and step information for iterative updates.

# Fields

- `∂Ω` : Boundary definition of the domain.
- `σ::AbstractVector` : Current conductivity values at all degrees of freedom.
- `δ::AbstractVector` : Proposed update to `σ` (gradient step, etc.).
- `L::Union{AbstractMatrix,Nothing}` : Current system matrix/operator for Neumann boundary problems.
- `L_fac` : Factorization of `L` for efficient solves.
- `LD::Union{AbstractMatrix,Nothing}` : Current system matrix/operator for Dirichlet boundary problems.
- `LD_fac` : Factorization of `LD`.
- `Σ::AbstractVector` : Singular values of system matrix K (for SVD or regularization purposes).
- `d` : Pseudo-metric: `(x,y) -> norm(x-y)^2` by default.
- `∂d` : Derivative of the pseudo-metric w.r.t. `x`: `(x,y) -> 2*(x-y)` by default.
- `n` : Pseudo-norm: `(x) -> norm(x)^2` by default.
- `∂n` : Derivative of the pseudo-norm: `(x) -> 2*x` by default.
- `R_diff` : Differentiable part of regularizer (callable function).
- `R_ndiff` : Non-differentiable convex part of regularizer.
- `∇R` : Gradient of the differentiable regularizer.
- `R_diff_args` : Arguments for the differentiable regularizer.
- `R_ndiff_args` : Arguments for the non-differentiable regularizer.
- `β_diff::Float64` : Regularization weight for differentiable part.
- `β_ndiff::Float64` : Regularization weight for non-differentiable part.
- `τ::Float64` : Learning rate for iterative updates.
- `num_pairs::Int64` : Number of voltage-current boundary pairs in the EIT experiment.
- `steps::Int64` : Number of steps taken in the solver iteration.
"""
mutable struct FerriteSolverState <: AbstractGalerkinSolver
    ∂Ω # Definition of the boundary
    σ::AbstractVector # Conductivity values
    δ::AbstractVector # Update of the conductivity
    L::Union{AbstractMatrix,Nothing} #Current guess of boundary operator for Neumann boundary
    L_fac # Factorized version of K
    LD::Union{AbstractMatrix,Nothing} # Current guess of boundary operator for dirichlet boundary
    LD_fac
    Σ::AbstractVector # Singular values of K
    d # pseudo-metric we use: d: (x, y) -> norm(x - y)^2 by default
    ∂d # derivative of the pseudo-metric after x: (x, y) -> 2 * (x - y) by default
    n # pseudo-norm we use: n: (x) -> norm(x)^2 by default
    ∂n # derivative of the pseudo-norm after x: (x) -> 2 * x by default
    R_diff # Some Function that holds the differentiable part of the regularizer
    R_ndiff # Some Function that holds the non-differentiable part of the regularizer required to be a convex lower-semicontinuous function
    ∇R # gradient of the regularizer
    R_diff_args # Arguments for the differentiable regularizer function
    R_ndiff_args # Arguments for the non-differentiable regularizer function
    β_diff::Float64 # Regularization parameter for differentiable part
    β_ndiff::Float64 # Regularization parameter for non-differentiable part
    τ::Float64 # Learn rate
    num_pairs::Int64 # Number of voltage-current pairs
    steps::Int64 # Number of steps taken
end

"""
    FerriteSolverState(fe::FerriteFESpace, σ::AbstractVector)

Constructs a `FerriteSolverState` with default pseudo-metric and pseudo-norm functions:
- `d(x,y) = norm(x - y)^2`
- `∂d(x,y) = 2*(x-y)`
- `n(x) = norm(x)^2`
- `∂n(x) = 2*x`

Initializes system matrices (`L`, `LD`), singular values, and update vector `δ` to zeros.
"""
function FerriteSolverState(fe::FerriteFESpace, σ::AbstractVector)
    d = (x, y) -> norm(x - y)^2
    ∂d = (x, y) -> 2 * (x - y)
    n = (x) -> norm(x)^2
    ∂n = (x) -> 2 * x
    FerriteSolverState(fe, σ, d, ∂d, n, ∂n)
end

makegradₓ(d) = (x, y) -> Enzyme.gradient(Reverse, Const(d), x, y)[1]
makegrad(n) = (x) -> Enzyme.gradient(Reverse, Const(n), x)

"""
    FerriteSolverState(fe::FerriteFESpace, σ::AbstractVector, d, n)

Constructs a `FerriteSolverState` where the pseudo-metric `d` and pseudo-norm `n` are user-defined.
Derivatives `∂d` and `∂n` are automatically created using Enzyme automatic differentiation.
"""
function FerriteSolverState(fe::FerriteFESpace, σ::AbstractVector, d, n)
    ∂d = makegradₓ(d)
    ∂n = makegrad(n)
    FerriteSolverState(fe, σ, d, ∂d, n, ∂n)
end

"""
    FerriteSolverState(fe::FerriteFESpace, σ::AbstractVector, d, ∂d, n, ∂n)

Constructs a `FerriteSolverState` with user-specified pseudo-metric, its derivative,
pseudo-norm, and its derivative. System matrices and boundary definitions are initialized
according to the provided finite element space `fe`.
"""
function FerriteSolverState(fe::FerriteFESpace, σ::AbstractVector, d, ∂d, n, ∂n)
    ∂Ω = fe.∂Ω
    δ = zeros(fe.n)

    L = assemble_L(fe, σ)
    Σ = zeros(fe.m - 1)

    FerriteSolverState(∂Ω, σ, δ, L, nothing, nothing, nothing, Σ, d, ∂d, n, ∂n, nothing,nothing, nothing, nothing, nothing, 0.0, 0.0, 0.1, 0, 0)
end
