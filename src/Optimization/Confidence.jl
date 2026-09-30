# Confidence maps: how well the data determine each parameter (pixel, coefficient), from the
# Jacobian SVD J = U S Vᵀ W (see TruncatedSVD.jl).
#
# - sensitivity:  ‖J eⱼ‖ / wⱼ, the raw sensitivity (per unit of the metric).
# - resolution:   the diagonal of the model resolution matrix R = V F Vᵀ W of a regularised
#                 linearized reconstruction, F = diag(fᵢ) the filter factors: fᵢ = 1 on the kept
#                 modes for truncation, fᵢ = sᵢ² / (sᵢ² + λ s₁²) for Tikhonov/LM damping in W.
#                 Rⱼⱼ = wⱼ Σᵢ fᵢ Vⱼᵢ² ∈ [0, 1]: 1 where the data determine the parameter, 0 where
#                 it comes from the initial guess or the prior.
# - posterior:    for the prior θ ~ N(θ₀, γ² W⁻¹) and residual noise N(0, η² I), the linearized
#                 posterior covariance C = (JᵀJ / η² + W / γ²)⁻¹ has
#                     Cⱼⱼ = (γ² / wⱼ) (1 - Rⱼⱼ)   with Tikhonov filter λ s₁² = η² / γ²,
#                 so the posterior variance is the prior variance times one minus the resolution.

"""
    sensitivity_map(obj, θ; weights = nothing)

Sensitivity of the data to every parameter at `θ`: the column norms `‖J eⱼ‖` of the Jacobian of
the least-squares objective `obj`, divided by `weights` (e.g. pixel areas or the lumped mass, to
get a density independent of the element size).
"""
function sensitivity_map(obj::AbstractObjective, θ::AbstractVector; weights = nothing)
    _require_least_squares(obj, "sensitivity_map")
    s = if _has_row_blocks(obj)
        jacobian_column_norms(obj, Vector{Float64}(θ))          # row blocks, J never stored
    else
        r = zeros(n_residual(obj))
        J = zeros(length(r), length(θ))
        residual_and_jacobian!(r, J, obj, Vector{Float64}(θ))
        vec(sqrt.(sum(abs2, J; dims = 1)))
    end
    weights === nothing || (s ./= weights)
    return s
end

"""
    resolution_map(obj, θ; rank = nothing, rtol = nothing, λ = nothing, weights = nothing)
    resolution_map(js; rank = nothing, rtol = nothing, λ = nothing)

Diagonal of the model resolution matrix `R = V F Vᵀ W` of the linearized reconstruction at `θ`,
in `[0, 1]`: `1` where the data determine the parameter, `0` where it is left to the initial
guess or prior. The filter `F` is either a truncation, with the modes of index `≤ rank` and
singular value `≥ rtol s₁` kept (as in [`TruncatedGaussNewton`](@ref)), or Tikhonov damping
`sᵢ² / (sᵢ² + λ s₁²)` (Levenberg–Marquardt with damping matrix `W`). `js` is a precomputed
[`jacobian_svd`](@ref).
"""
function resolution_map(obj::AbstractObjective, θ::AbstractVector; rank = nothing, rtol = nothing, λ = nothing,
                        weights = nothing)
    return resolution_map(jacobian_svd(obj, θ; weights); rank, rtol, λ)
end

function resolution_map(js::NamedTuple; rank = nothing, rtol = nothing, λ = nothing)
    f = _filter_factors(js.s; rank, rtol, λ)
    return js.w .* (abs2.(js.V) * f)
end

function _filter_factors(s; rank, rtol, λ)
    truncation = rank !== nothing || rtol !== nothing
    truncation == (λ === nothing) ||
        throw(ArgumentError("give either a truncation (rank and/or rtol) or a Tikhonov λ"))
    isempty(s) && return Float64[]
    if truncation
        rank === nothing || rank >= 1 || throw(ArgumentError("rank must be positive, got $rank"))
        rtol === nothing || 0 <= rtol < 1 || throw(ArgumentError("rtol must lie in [0, 1), got $rtol"))
        k = min(rank === nothing ? length(s) : rank, count(>=((rtol === nothing ? 0.0 : rtol) * s[1]), s),
                count(>(0), s))
        return [i <= k ? 1.0 : 0.0 for i in eachindex(s)]
    end
    λ >= 0 || throw(ArgumentError("λ must be nonnegative"))
    return abs2.(s) ./ (abs2.(s) .+ λ * s[1]^2)
end

"""
    posterior_std(obj, θ; noise, prior_std, weights = nothing)
    posterior_std(js; noise, prior_std)

Pointwise standard deviation of the linearized Gaussian posterior at `θ`, for the prior
`θ ~ N(θ₀, prior_std² W⁻¹)` (`W = Diagonal(weights)`, identity by default: independent parameters
with standard deviation `prior_std`) and residual noise `N(0, η² I)`:
`sqrt.(diag((JᵀJ / η² + W / prior_std²)⁻¹))`. It equals the prior standard deviation times
`sqrt(1 - R)`, `R` the Tikhonov [`resolution_map`](@ref) with `λ s₁² = η² / prior_std²`.

`noise` is `η` in units of the (whitened) residual, or a noise model
([`RelativeGaussianNoise`](@ref), [`GaussianNoise`](@ref)) for an [`AdjointStateObjective`](@ref)
(possibly parametrized), converted to the mean residual variance
`η² = 2 discrepancy_target(obj, noise; τ = 1) / n_residual(obj)`.
"""
function posterior_std(obj::AbstractObjective, θ::AbstractVector; noise, prior_std::Real, weights = nothing)
    return posterior_std(jacobian_svd(obj, θ; weights); noise = _residual_noise_std(obj, noise), prior_std)
end

function posterior_std(js::NamedTuple; noise::Real, prior_std::Real)
    noise > 0 && prior_std > 0 || throw(ArgumentError("noise and prior_std must be positive"))
    R = resolution_map(js; λ = isempty(js.s) || js.s[1] == 0 ? 0.0 : (noise / (prior_std * js.s[1]))^2)
    return prior_std .* sqrt.(max.(1 .- R, 0) ./ js.w)
end

_residual_noise_std(obj, η::Real) = Float64(η)
_residual_noise_std(o::ParametrizedObjective, noise::AbstractNoiseModel) = _residual_noise_std(o.obj, noise)
_residual_noise_std(obj, noise::AbstractNoiseModel) =
    throw(ArgumentError("noise models are supported for AdjointStateObjective; give the residual noise level η instead"))
