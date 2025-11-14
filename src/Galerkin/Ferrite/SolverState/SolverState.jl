using Ferrite
using Enzyme
using LinearMaps

export FerriteSolverState
export GalerkinOptState
export add_diff_Regularizer!
export add_ndiff_Regularizer!
export update_L!
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
mutable struct GalerkinOptState
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
    fe::FerriteFESpace # Finite element space
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
    H_reg # Hessian of the regularizer
    R_diff_args # Arguments for the differentiable regularizer function
    R_ndiff_args # Arguments for the non-differentiable regularizer function
    num_pairs::Int64 # Number of voltage-current pairs
    opt::GalerkinOptState # State of the optimization algorithm
    modes
    clip::Bool
    clip_value::Float64
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
    opt = GalerkinOptState(nothing, nothing, 0.0, 0.0, 0.1, 0, 1e-5, nothing, copy(δ))
    FerriteSolverState(fe, ∂Ω, σ, δ, L, nothing, nothing, nothing, Σ, d, ∂d, n, ∂n, nothing, nothing, nothing, nothing, nothing, nothing, 0, opt, nothing, false, 1.0)
end


function update_sigma!(state::FerriteSolverState, clip::Bool=false, clip_limit::Float64=1.0)
    state.σ .= min.(state.σ .+ state.opt.τ .* state.δ, 1e-6)
    if clip
        state.σ .= max.(state.σ, clip_limit)
    end
end

function update_L!(state::FerriteSolverState)
    state.L .= assemble_L(state.L, state.fe, state.σ)
end

function add_diff_Regularizer!(state::FerriteSolverState, Reg, R_diff_args, ∇Reg)
    state.R_diff = Reg
    state.R_diff_args = R_diff_args
    state.∇R = ∇Reg
end
function add_diff_Regularizer!(state::FerriteSolverState, Reg, R_diff_args)
    state.R_diff = Reg
    state.R_diff_args = R_diff_args
    state.∇R = Enzyme.gradient(Reg)
end
function add_ndiff_Regularizer!(state::FerriteSolverState, nReg, R_ndiff_args)
    state.R_ndiff = nReg
    state.R_ndiff_args = R_ndiff_args
end


function init_opt!(state::FerriteSolverState, n::Int)
    state.opt.J = zeros(n, state.fe.n)
    state.opt.r = zeros(state.fe.n)
    state.op.λ = 1e-5
end
