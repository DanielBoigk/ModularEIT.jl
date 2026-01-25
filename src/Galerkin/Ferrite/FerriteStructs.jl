using Ferrite
using SparseArrays
using Enzyme
using LinearMaps

export FerriteOptState
export FerriteProblem
export FerriteSolverState
export FerriteFESpace
export FerriteEITMode


"""
    mutable struct GalerkinOptState

Holds the optimization state for a Gauss–Newton–type solver in a Galerkin or finite element setting.

# Fields
- `J::Union{AbstractMatrix,Nothing}` — Current Jacobian matrix of the forward map with respect to the parameters `σ`.
- `r::Union{AbstractVector,Nothing}` — Residual vector between simulated and measured boundary data.
- `β_diff::Float64` — Regularization parameter for the differentiable part of the regularizer.
- `β_ndiff::Float64` — Regularization parameter for the non-differentiable part of the regularizer.
- `τ::Float64` — Learning rate or step-size multiplier.
- `steps::Int` — Number of optimization steps performed.
- `λ::Float64` — Levenberg–Marquardt damping parameter controlling the tradeoff between Gauss–Newton and gradient descent behavior.
- `L::Union{AbstractMatrix,Nothing,LinearMap}` — Regularization (or prior precision) operator.
- `δ::AbstractVector` — Current update direction for the parameter vector `σ`.

This struct can be used both with dense or sparse matrices, and supports using `LinearMap`s to avoid forming `J` or `L` explicitly.
"""
mutable struct FerriteOptState <: GalerkinOptState
    J::Union{AbstractMatrix,Nothing} # Jacobian matrix
    r::Union{AbstractVector,Nothing} # Residual vector
    β_diff::Float64 # regularization parameter for differentiable part
    β_ndiff::Float64 # regularization parameter for non-differentiable part
    τ::Float64 # Learning rate
    steps::Int
    λ::Float64 # Levenberg-Marquardt parameter
    L::Union{AbstractMatrix,Nothing,LinearMap} # Regularization matrix
    δ::AbstractVector # Proposed update to `σ`
end

"""
    struct FerriteFESpace{RefElem} <: AbstractHilbertSpace

Finite Element Space for a given reference element type `RefElem`.

# Fields
- `cellvalues::CellValues` : precomputed shape functions and quadrature for elements.
- `dh::DofHandler` : mapping between local and global degrees of freedom.
- `∂Ω` : boundary faces (indices) for Dirichlet conditions.
- `facetvalues::FacetValues` : shape functions for facets used in boundary integrals.
- `ch::ConstraintHandler` : handles Dirichlet (or other) constraints.
- `order::Int` : polynomial order of the FE basis.
- `qr_order::Int` : quadrature order.
- `dim::Int` : spatial dimension.
- `n::Int` : number of global degrees of freedom.
- `num_facet::Int` : number of facets.
- `m::Int` : number of boundary facets.
- `M::AbstractMatrix` : mass matrix.
- `M_fac` : factorization of mass matrix (optional).
- `K::AbstractMatrix` : stiffness matrix.
- `K_fac` : factorization of stiffness matrix (optional).
- `total_volume::Float64` : total volume of the domain.
- `down` : projection from force vector to boundary coefficients.
- `up` : projection from boundary coefficients to force vector.
- `up!` : in-place version of `up`.
"""
struct FerriteFESpace{RefElem} <: AbstractHilbertSpace
    cellvalues::CellValues
    dh::DofHandler
    ∂Ω
    facetvalues::FacetValues
    ch::ConstraintHandler
    order::Int
    qr_order::Int
    dim::Int
    n::Int
    num_facet::Int
    m::Int
    M::AbstractMatrix
    M_fac
    K::AbstractMatrix
    K_fac
    total_volume::Float64
    down # Projection from force vector to coefficients of the basis functions of boundary
    up # Projection from coefficients of the basis functions of boundary to force vector
    up! # Projection from coefficients of the basis functions of boundary to force vector
end



"""
    mutable struct FerriteSolverState <: AbstractGalerkinSolver

Encapsulates the state of an EIT solver using the Ferrite FEM framework.
Holds the conductivity, PDE operators, regularizers, and step information for iterative updates.

# Fields

- `fe::FerriteFESpace` : Finite element space
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
    δ_updated::Bool
    error::Float64
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
    H_reg # Hessian of the regularizer
    R_diff_args # Arguments for the differentiable regularizer function
    R_ndiff_args # Arguments for the non-differentiable regularizer function
    num_pairs::Int64 # Number of voltage-current pairs
    N::Int64 # dimension of the boundary vector
    opt::FerriteOptState # State of the optimization algorithm
    clip::Bool
    clip_value::Float64
end


mutable struct FerriteEITMode
    u_f::Union{AbstractVector,Nothing}
    u_g::Union{AbstractVector,Nothing}
    w::Union{AbstractVector,Nothing}
    b::Union{AbstractVector,Nothing}
    λ::AbstractVector
    δσ::AbstractVector
    F::Union{AbstractVector,Nothing} # This is the long vector for dirichlet boundary conditions
    f::Union{AbstractVector,Nothing} # This is the short vector for dirichlet boundary conditions
    G::Union{AbstractVector,Nothing} # This is the long vector for neumann boundary conditions
    g::Union{AbstractVector,Nothing} # This is the short vector for neumann boundary conditions
    λrhs::AbstractVector
    rhs::AbstractVector # This is a preallocation for calculating the bilinear map
    error_d::Float64
    error_n::Float64
    error_m::Float64
end

export FerriteProblem

struct FerriteProblem
    fe::FerriteFESpace
    modes::Dict{Int64,FerriteEITMode}
    state::FerriteSolverState
end
