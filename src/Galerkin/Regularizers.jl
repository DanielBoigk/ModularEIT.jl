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

`ε = 0` gives the exact (non-smooth) total variation. Its proximal operator ([`prox!`](@ref)) is
computed exactly by the Chambolle–Pock method, but it has no gradient where σ is constant, so
use it with [`ProximalGradient`](@ref) or [`ADMM`](@ref) rather than with gradient-based methods.
"""
struct TotalVariationRegularizer{D <: AbstractDiscretization, C} <: AbstractRegularizer
    disc::D
    ε::Float64
    cache::C                          # back-end data, e.g. the facet graph of piecewise constants
    prox_state::Base.RefValue{Any}    # operator and warm start of the exact prox (built on first use)
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

# ---------------------------------------------------------------------------------------------
# Proximal operators
#
#     prox(v) = argmin_{lower ≤ z ≤ upper}  R(z) + ρ/2 Σᵢ wᵢ (zᵢ - vᵢ)²
#
# in a diagonal metric w (w = 1: Euclidean; w = lumped σ mass matrix: discrete L², independent of
# the mesh). Diagonal metrics keep the prox of box constraints a clamp and the prox of TV a
# Chambolle–Pock iteration with closed-form steps. Generic fallback for smooth regularizers: the
# lagged Newton iteration (H(z) + ρW) z⁺ = ρ W v + H(z) z - ∇R(z) with the Gauss–Newton Hessian
# model (one step for quadratics, lagged diffusivity for smoothed TV); with bounds, projected
# L-BFGS on the (PDE-free) prox problem.

"""
    prox!(z, reg, v, ρ; weights = nothing, lower = nothing, upper = nothing, tol = nothing,
          maxiter = nothing)
    prox(reg, v, ρ; kwargs...)

Proximal operator `argmin R(z) + ρ/2 Σᵢ wᵢ (zᵢ - vᵢ)²` subject to `lower ≤ z ≤ upper`, in the
diagonal metric `w = weights` (default: Euclidean; pass [`lumped_mass`](@ref) for the discrete
L² metric). Exact for [`TikhonovRegularizer`](@ref) without bounds and for the non-smooth
[`TotalVariationRegularizer`](@ref) with `ε = 0` (Chambolle–Pock); iterative for other smooth
regularizers. A [`ProximalMap`](@ref) wraps user-defined maps such as denoisers.

For the iterative TV prox, `tol` is an absolute tolerance on the primal–dual gap, which bounds
`ρ/2 ‖z - z*‖²_w` (default: round-off level), and `maxiter` caps the iterations. The proximal
methods pass tolerances tied to their own progress; other proximal maps ignore both.
"""
function prox!(z::AbstractVector, reg::AbstractRegularizer, v::AbstractVector, ρ::Real;
               weights = nothing, lower = nothing, upper = nothing, tol = nothing, maxiter = nothing)
    ρ > 0 || throw(ArgumentError("ρ must be positive"))
    w = weights === nothing ? ones(length(v)) : weights
    if lower === nothing && upper === nothing
        return _lagged_newton_prox!(z, reg, v, ρ, w)
    end
    return _bounded_smooth_prox!(z, reg, v, ρ, w, lower, upper)
end
prox(reg::AbstractRegularizer, v::AbstractVector, ρ::Real; kwargs...) = prox!(similar(v, Float64), reg, v, ρ; kwargs...)

function _lagged_newton_prox!(z, reg, v, ρ, w; maxiter = 200, rtol = 1e-12)
    n = length(v)
    copyto!(z, v)
    g = zeros(n)
    W = spdiagm(0 => ρ .* w)
    for _ in 1:maxiter
        value_and_gradient!(g, reg, z)
        H = gauss_newton_hessian(reg, z)
        b = ρ .* w .* v .+ H * z .- g
        znew = cholesky(Symmetric(sparse(H) + W)) \ b
        δ = norm(znew .- z, Inf)
        copyto!(z, znew)
        δ <= rtol * (1 + norm(z, Inf)) && break
    end
    return z
end

# the prox problem as an objective (no PDE), for projected L-BFGS
struct _ProxObjective{R, V} <: AbstractObjective
    reg::R
    v::V
    ρ::Float64
    w::Vector{Float64}
end
objective_value(p::_ProxObjective, z::AbstractVector) =
    objective_value(p.reg, z) + p.ρ / 2 * sum(i -> p.w[i] * (z[i] - p.v[i])^2, eachindex(z))
function value_and_gradient!(g::AbstractVector, p::_ProxObjective, z::AbstractVector)
    R = value_and_gradient!(g, p.reg, z)
    g .+= p.ρ .* p.w .* (z .- p.v)
    return R + p.ρ / 2 * sum(i -> p.w[i] * (z[i] - p.v[i])^2, eachindex(z))
end

# Riesz map by a diagonal scaling (the inverse metric)
struct _DiagonalRiesz <: AbstractRieszMap
    d::Vector{Float64}
end
riesz_map!(g::AbstractVector, R::_DiagonalRiesz) = (g .*= R.d; g)

function _bounded_smooth_prox!(z, reg, v, ρ, w, lower, upper)
    obj = _ProxObjective(reg, v, Float64(ρ), Vector{Float64}(w))
    res = minimize(obj, v, LBFGS(; riesz = _DiagonalRiesz(1 ./ (ρ .* w))); lower, upper,
                   maxiter = 2000, gtol = 1e-12)
    return copyto!(z, res.σ)
end

"""
    ProximalMap(f!)

A regularizer given only by its proximal map `f!(z, v, ρ)` (write `argmin R(z) + ρ/2 ‖z - v‖²`
into `z`), e.g. a denoiser for plug-and-play priors. Bounds are applied by clamping afterwards
and the metric weights are ignored. Its value is reported as 0.
"""
struct ProximalMap{F} <: AbstractRegularizer
    f!::F
end
objective_value(::ProximalMap, σ::AbstractVector) = 0.0
function prox!(z::AbstractVector, pm::ProximalMap, v::AbstractVector, ρ::Real;
               weights = nothing, lower = nothing, upper = nothing, tol = nothing, maxiter = nothing)
    pm.f!(z, v, ρ)
    lower === nothing || (z .= max.(z, lower))
    upper === nothing || (z .= min.(z, upper))
    return z
end
