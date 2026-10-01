# Abstract interfaces shared by all Galerkin back ends (Ferrite.jl now; Gridap.jl and
# GalerkinToolkit.jl later). Back ends implement concrete subtypes; objectives, electrode models
# and linear solvers only talk to these interfaces.

"""
    AbstractDiscretization

Discrete function spaces for the potential `u` and the conductivity `σ` over one mesh. `u` and `σ`
live in different finite element spaces on the same grid (e.g. P1/P0, but any pair is allowed).
Carries the boundary description and everything needed to assemble mass, stiffness and weighted
stiffness matrices. After adaptive mesh refinement a new discretization is built.

Interface: [`ndofs_u`](@ref), [`ndofs_σ`](@ref).
"""
abstract type AbstractDiscretization end

"""
    AbstractElectrodeModel

How current enters and voltage is measured on the boundary (continuum, point, gap or complete
electrode model). A [`ForwardModel`](@ref) turns an electrode model on a discretization into
injection and measurement matrices.
"""
abstract type AbstractElectrodeModel end

"""
    AbstractForwardModel

A discretized EIT forward problem: system matrix `A(σ) = A₀ + Σₐ σₐ ∂A/∂σₐ`, injection matrix
`P`, measurement matrix `Q`, null space and grounding of the current-driven problem, and the
structure of the voltage-driven (Dirichlet) problem.
"""
abstract type AbstractForwardModel end

"""
    AbstractObjective

A reconstruction functional `J(σ)` with gradient, e.g. the least-squares data misfit through the
adjoint state method ([`AdjointStateObjective`](@ref)) or the Kohn–Vogelius functional
([`KohnVogeliusObjective`](@ref)). Interface: [`objective_value`](@ref),
[`value_and_gradient!`](@ref).
"""
abstract type AbstractObjective end

"""
    AbstractMisfit

Metric in which the data misfit is measured. `J = ½ ‖U e‖²` for a whitening operator `U`,
e.g. [`SquaredEuclidean`](@ref) (`U = I`) or [`WeightedSquaredEuclidean`](@ref) (`UᵀU = W`).
"""
abstract type AbstractMisfit end

"""
    AbstractRieszMap

Representation of the gradient: the coefficient gradient `∂J/∂σₐ` of the discrete functional
(discretize-then-optimize, [`CoefficientGradient`](@ref)) or its Riesz representative in a
function space, e.g. the L² gradient `M_σ⁻¹ ∂J/∂σ` ([`L2Gradient`](@ref)).
"""
abstract type AbstractRieszMap end

"""
    AbstractRegularizer

A regularization functional `R(σ)` with value, coefficient gradient and a Gauss–Newton Hessian
model, e.g. [`TikhonovRegularizer`](@ref) or [`TotalVariationRegularizer`](@ref). Added to a
data objective by [`RegularizedObjective`](@ref).
"""
abstract type AbstractRegularizer end

"""
    AbstractOptimizer

Minimization method for [`minimize`](@ref): [`GradientDescent`](@ref), [`LBFGS`](@ref),
[`GaussNewton`](@ref).
"""
abstract type AbstractOptimizer end

"""
    AbstractNoiseModel

Measurement noise added to simulated data: [`GaussianNoise`](@ref),
[`RelativeGaussianNoise`](@ref), [`SourceMeterNoise`](@ref). See [`add_noise`](@ref) and
[`simulate_data`](@ref).
"""
abstract type AbstractNoiseModel end

"""
    AbstractInclusion

A shape with a conductivity value for [`InclusionPhantom`](@ref): [`CircleInclusion`](@ref),
[`EllipseInclusion`](@ref), [`PolygonInclusion`](@ref). Membership: `x in inclusion`.
"""
abstract type AbstractInclusion end

"""
    AbstractParametrization

A linear map `σ = P θ` from reconstruction parameters to conductivity coefficients, e.g. pixel
values (`PixelParametrization` (Ferrite back end)) or subspaces of them ([`SubspaceParametrization`](@ref)).
Objectives in the parameters: [`ParametrizedObjective`](@ref).
"""
abstract type AbstractParametrization end

"""
    AbstractLinearSolver

Choice of linear solver for the state, adjoint and Dirichlet systems:
[`DirectSolver`](@ref) (projected sparse Cholesky) or [`BlockCGSolver`](@ref).
"""
abstract type AbstractLinearSolver end

"""
    AbstractEITProblem

A complete reconstruction problem (discretization, forward model, data, objective, current
iterate), solved by [`reconstruct!`](@ref); see [`EITProblem`](@ref).
"""
abstract type AbstractEITProblem end

"""
    AbstractSolutionState

State of an iterative reconstruction (iterate, step sizes, histories), e.g. the
[`OptimizationState`](@ref) returned by [`minimize`](@ref).
"""
abstract type AbstractSolutionState end
