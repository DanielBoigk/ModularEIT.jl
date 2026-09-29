# Singular value decomposition of the Jacobian, and Gauss–Newton with truncated-SVD steps.
#
# In a metric W = diag(w) on the parameters (e.g. the lumped σ mass for finite element
# coefficients), the SVD of Ĵ = J W^{-1/2} = U S Ṽᵀ gives
#
#     J = U S Vᵀ W,     V = W^{-1/2} Ṽ,     Vᵀ W V = I:
#
# parameter modes vᵢ, orthonormal in W, and data modes uᵢ, ordered by the singular values sᵢ, i.e.
# by how well the measurements determine each mode. For EIT they are ordered by depth: the leading
# modes live near the boundary, the trailing ones in the interior.
#
# Truncated Gauss–Newton replaces the Gauss–Newton step by the W-minimum-norm step on the leading
# k modes,
#
#     δ = -Σ_{i ≤ k} (uᵢᵀ r / sᵢ) vᵢ,
#
# regularisation by projection onto the well-determined directions of the current Jacobian,
# without a penalty term (in contrast to Levenberg–Marquardt with identity damping, which filters
# the same modes smoothly by sᵢ² / (sᵢ² + λ)). The step is followed by projected backtracking,
# and iterating to the discrepancy principle (`ftarget`) is the usual stopping rule.

"""
    jacobian_svd(obj, θ; weights = nothing)

Singular value decomposition `J = U S Vᵀ W` of the Jacobian of the least-squares objective `obj`
at `θ` (thin: `min(m, n)` singular values, in decreasing order). The columns of `V` are the
parameter modes, orthonormal in the metric `W = Diagonal(weights)` (Euclidean by default), ordered
by how well the data determine them. Returns a named tuple `(U, s, V, r)` with the residual `r`.

For a [`ParametrizedObjective`](@ref) over pixels, `V[:, 1:k]` is the data-optimal subspace of
dimension `k` (see [`jacobian_basis`](@ref)).
"""
function jacobian_svd(obj::AbstractObjective, θ::AbstractVector; weights = nothing)
    _require_least_squares(obj, "jacobian_svd")
    r = zeros(n_residual(obj))
    J = zeros(length(r), length(θ))
    residual_and_jacobian!(r, J, obj, Vector{Float64}(θ))
    U, s, V = _weighted_svd(J, weights)
    return (; U, s, V, r)
end

"""
    jacobian_basis(obj, θ, k; weights = nothing)

The `k` leading parameter modes `V[:, 1:k]` of [`jacobian_svd`](@ref): the subspace best
determined by the data at `θ`, e.g. for `SubspaceParametrization(pixels, jacobian_basis(obj, θ, k))`.
"""
function jacobian_basis(obj::AbstractObjective, θ::AbstractVector, k::Integer; weights = nothing)
    js = jacobian_svd(obj, θ; weights)
    1 <= k <= length(js.s) || throw(ArgumentError("k must lie in 1:$(length(js.s)), got $k"))
    return js.V[:, 1:k]
end

function _require_least_squares(obj, name)
    hasmethod(residual_and_jacobian!, Tuple{Vector{Float64}, Matrix{Float64}, typeof(obj), Vector{Float64}}) &&
        hasmethod(n_residual, Tuple{typeof(obj)}) ||
        throw(ArgumentError("$name needs a least-squares objective with residual_and_jacobian! and n_residual " *
                            "(e.g. AdjointStateObjective), got $(nameof(typeof(obj)))"))
    return nothing
end

# J = U S Vᵀ W with Vᵀ W V = I
function _weighted_svd(J::AbstractMatrix, weights)
    if weights === nothing
        F = svd(J)
        return F.U, F.S, F.V
    end
    length(weights) == size(J, 2) || throw(DimensionMismatch("weights need length $(size(J, 2))"))
    all(>(0), weights) || throw(ArgumentError("weights must be positive"))
    sq = sqrt.(weights)
    F = svd(J ./ sq')
    return F.U, F.S, F.V ./ sq
end

"""
    TruncatedGaussNewton(; rank = nothing, rtol = nothing, weights = nothing)

Gauss–Newton method with truncated-SVD steps: at each iterate, the step uses only the leading
singular modes of the Jacobian (see [`jacobian_svd`](@ref)), those with index `≤ rank` and
singular value `≥ rtol · s₁` (at least one of the two must be given). The step is the
minimum-norm step in the metric `Diagonal(weights)`, followed by projected backtracking; bounds
are handled on the free set.

Truncation is the regulariser: the objective must be a pure least-squares objective (no
[`RegularizedObjective`](@ref)). Stop by the discrepancy principle, `ftarget =
discrepancy_target(obj, noise)`.
"""
struct TruncatedGaussNewton{W} <: AbstractOptimizer
    rank::Int
    rtol::Float64
    weights::W
end

function TruncatedGaussNewton(; rank = nothing, rtol = nothing, weights = nothing)
    rank === nothing && rtol === nothing && throw(ArgumentError("give a rank, an rtol or both"))
    rank === nothing || rank >= 1 || throw(ArgumentError("rank must be positive, got $rank"))
    rtol === nothing || 0 <= rtol < 1 || throw(ArgumentError("rtol must lie in [0, 1), got $rtol"))
    weights === nothing || all(>(0), weights) || throw(ArgumentError("weights must be positive"))
    return TruncatedGaussNewton(rank === nothing ? typemax(Int) : Int(rank),
                                rtol === nothing ? 0.0 : Float64(rtol), weights)
end

function _workspace(m::TruncatedGaussNewton, obj, n)
    obj isa RegularizedObjective &&
        throw(ArgumentError("TruncatedGaussNewton regularizes by truncation; use a pure least-squares objective " *
                            "(or GaussNewton for penalties)"))
    _require_least_squares(obj, "TruncatedGaussNewton")
    m.weights === nothing || length(m.weights) == n || throw(DimensionMismatch("weights need length $n"))
    return _workspace(GaussNewton(; damping = :linesearch), obj, n)
end

_initialize!(st, ws::_GaussNewtonWorkspace, m::TruncatedGaussNewton, obj, box) =
    _initialize!(st, ws, GaussNewton(; damping = :linesearch), obj, box)

function _step!(st, ws::_GaussNewtonWorkspace, m::TruncatedGaussNewton, obj, box)
    gnorm = _projected_gradient_norm(st.σ, st.g, box)
    _binding!(ws.mask, st.σ, st.g, box, _binding_tolerance(st.σ, gnorm))
    F = findall(!, ws.mask)
    isempty(F) && return false
    U, s, V = _weighted_svd(ws.Jm[:, F], m.weights === nothing ? nothing : m.weights[F])
    isempty(s) || s[1] > 0 || return false
    k = min(m.rank, count(>=(m.rtol * s[1]), s), count(>(0), s))
    c = (U[:, 1:k]' * ws.r) ./ s[1:k]
    fill!(ws.δ, 0)
    ws.δ[F] .= V[:, 1:k] * c                          # σ(t) = P(σ - t δ): δ is minus the step
    t = _projected_backtracking!(st, obj, box, ws.δ, 1.0, ws.σt, ws.gt)
    t > 0 || return false
    _linearize!(st, ws, obj)
    return true
end
