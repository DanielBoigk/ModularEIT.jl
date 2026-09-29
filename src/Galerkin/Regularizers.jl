# Regularizers R(σ) and regularized objectives J(σ) = J_data(σ) + Σₖ αₖ Rₖ(σ).
#
# A regularizer provides its value, its coefficient gradient and a symmetric positive
# semidefinite Hessian model for the Gauss–Newton methods (the exact Hessian for quadratic
# regularizers, the lagged-diffusivity matrix for total variation). The types here are
# independent of the finite element package; the constructors that build Gram matrices or facet
# graphs from a discretization live with the back end (Ferrite/Regularizers.jl).
#
# Gradients are always coefficient gradients (dual vectors). The Riesz map that turns them into
# a descent direction belongs to the optimiser, so that data term and regularizers are added in
# the same representation and the curvature pairs of quasi-Newton methods are dual pairings.

"""
    TikhonovRegularizer(G; reference = 0)

Quadratic regularizer `R(σ) = ½ (σ - σ₀)ᵀ G (σ - σ₀)` for a symmetric positive semidefinite
Gram matrix `G` and a reference conductivity `σ₀` (scalar or vector). With a discretization,
`TikhonovRegularizer(disc; kind, reference)` assembles `G` for the `:L2`, `:H1semi`, `:H1`
or `:jump` (piecewise constants) norm.

Interface: [`objective_value`](@ref), [`value_and_gradient!`](@ref),
[`gauss_newton_hessian`](@ref).
"""
struct TikhonovRegularizer{MG <: AbstractMatrix, V <: AbstractVector} <: AbstractRegularizer
    G::MG
    reference::V
    buffer::V
end

function TikhonovRegularizer(G::AbstractMatrix; reference = 0.0)
    n = LinearAlgebra.checksquare(G)
    σ0 = reference isa Real ? fill(Float64(reference), n) : Vector{Float64}(reference)
    length(σ0) == n || throw(DimensionMismatch("reference needs $n entries"))
    return TikhonovRegularizer(G, σ0, zeros(n))
end

function objective_value(reg::TikhonovRegularizer, σ::AbstractVector)
    d = reg.buffer
    d .= σ .- reg.reference
    return dot(d, reg.G, d) / 2
end

function value_and_gradient!(g::AbstractVector, reg::TikhonovRegularizer, σ::AbstractVector)
    d = reg.buffer
    d .= σ .- reg.reference
    mul!(g, reg.G, d)
    return dot(d, g) / 2
end

"""
    gauss_newton_hessian(reg, σ)

Symmetric positive semidefinite Hessian model of the regularizer at `σ` (sparse), used by
[`GaussNewton`](@ref): the Gram matrix for [`TikhonovRegularizer`](@ref), the lagged
diffusivity matrix `H(σ)` with `∇R(σ) = H(σ) σ` for [`TotalVariationRegularizer`](@ref).
"""
gauss_newton_hessian(reg::TikhonovRegularizer, σ::AbstractVector) = reg.G

"""
    TotalVariationRegularizer(disc; ε = 1e-3)

Smoothed total variation `R(σ) = TV_ε(σ)` of the conductivity (see [`total_variation`](@ref)):
facet jumps `Σ_F |F| √((σ_K - σ_K')² + ε²)` for piecewise constants, `∫ √(|∇σ|² + ε²)` for
continuous σ. The Gauss–Newton Hessian model is the lagged diffusivity matrix (the jump or
stiffness matrix weighted by `1/√(… + ε²)` at the current σ).
"""
struct TotalVariationRegularizer{D <: AbstractDiscretization, C} <: AbstractRegularizer
    disc::D
    ε::Float64
    cache::C          # back-end data, e.g. the facet graph of piecewise constants
end

"""
    RegularizedObjective(data, α₁ => R₁, α₂ => R₂, …)

`J(σ) = J_data(σ) + Σₖ αₖ Rₖ(σ)` for a data objective (e.g. [`AdjointStateObjective`](@ref),
[`KohnVogeliusObjective`](@ref)) and regularizers `Rₖ` with weights `αₖ ≥ 0`. The data
objective must deliver coefficient gradients (`gradient = CoefficientGradient()`); choose the
gradient representation in the optimiser instead (e.g. `LBFGS(riesz = L2Gradient(mats))`).
[`GaussNewton`](@ref) needs a least-squares data objective with residuals and Jacobians.
"""
struct RegularizedObjective{O <: AbstractObjective, R <: Tuple} <: AbstractObjective
    data::O
    weights::Vector{Float64}
    regularizers::R
    buffer::Vector{Float64}
end

function RegularizedObjective(data::AbstractObjective, terms::Pair{<:Real, <:AbstractRegularizer}...)
    _require_coefficient_gradient(data)
    weights = Float64[first(t) for t in terms]
    all(>=(0), weights) || throw(ArgumentError("regularization weights must be nonnegative"))
    return RegularizedObjective(data, weights, map(last, terms), Float64[])
end

function objective_value(obj::RegularizedObjective, σ::AbstractVector)
    J = objective_value(obj.data, σ)
    for (α, reg) in zip(obj.weights, obj.regularizers)
        J += α * objective_value(reg, σ)
    end
    return J
end

function value_and_gradient!(g::AbstractVector, obj::RegularizedObjective, σ::AbstractVector)
    J = value_and_gradient!(g, obj.data, σ)
    length(obj.buffer) == length(g) || resize!(obj.buffer, length(g))
    gr = obj.buffer
    for (α, reg) in zip(obj.weights, obj.regularizers)
        J += α * value_and_gradient!(gr, reg, σ)
        g .+= α .* gr
    end
    return J
end

# Representation of the gradients an objective delivers. Optimisers and regularized objectives
# require coefficient gradients.
_gradient_representation(::AbstractObjective) = CoefficientGradient()
_gradient_representation(obj::RegularizedObjective) = _gradient_representation(obj.data)

function _require_coefficient_gradient(obj::AbstractObjective)
    _gradient_representation(obj) isa CoefficientGradient && return nothing
    throw(ArgumentError("the objective must deliver coefficient gradients (gradient = CoefficientGradient()); " *
                        "pass the Riesz map to the optimiser instead, e.g. LBFGS(riesz = L2Gradient(mats))"))
end
