# Data misfit metrics. A misfit is J = ½ ‖U e‖² with a whitening operator U applied to each
# column (pattern) of the error e; U = I is the squared Euclidean distance. The residual of the
# least-squares problem is r = U e, so Gauss–Newton and gradient methods see the same metric.

"""
    SquaredEuclidean()

`J = ½ Σₛ ‖eₛ‖²` over the patterns `s`.
"""
struct SquaredEuclidean <: AbstractMisfit end

"""
    WeightedSquaredEuclidean(W)

`J = ½ Σₛ eₛᵀ W eₛ` for a symmetric positive definite `W` (`n_obs × n_obs`), e.g. inverse noise
covariances or a boundary mass matrix. Whitened residual `r = U e` with `UᵀU = W`.
"""
struct WeightedSquaredEuclidean{MW, MU} <: AbstractMisfit
    W::MW
    U::MU
end
WeightedSquaredEuclidean(W::Diagonal) = WeightedSquaredEuclidean(W, Diagonal(sqrt.(W.diag)))
WeightedSquaredEuclidean(W::AbstractMatrix) = WeightedSquaredEuclidean(W, cholesky(Symmetric(Matrix(W))).U)

"""
    ProjectedMisfit(U)

`J = ½ Σₛ ‖U eₛ‖²` for a rectangular `U` (`k × n_obs`, `k ≤ n_obs`): only `k` combinations of the
measurements enter the misfit, e.g. the leading measurement modes of [`pattern_svd`](@ref)
(`truncate_patterns(p, K; measurements = M)`, a two-sided truncation of the data with `K M`
residuals instead of `n_obs K`). The residual has `k` entries per pattern.
"""
struct ProjectedMisfit{MU <: AbstractMatrix} <: AbstractMisfit
    U::MU
end

# number of residual entries per pattern
_residual_rows(::AbstractMisfit, n_obs) = n_obs
function _residual_rows(m::ProjectedMisfit, n_obs)
    size(m.U, 2) == n_obs || throw(DimensionMismatch("the projection needs $n_obs columns, got $(size(m.U, 2))"))
    return size(m.U, 1)
end

# r ← U e
_whiten!(R, ::SquaredEuclidean, E) = copyto!(R, E)
_whiten!(R, m::WeightedSquaredEuclidean, E) = mul!(R, m.U, E)
_whiten!(R, m::ProjectedMisfit, E) = mul!(R, m.U, E)
# G ← Uᵀ R
_whiten_adjoint!(G, ::SquaredEuclidean, R) = copyto!(G, R)
_whiten_adjoint!(G, m::WeightedSquaredEuclidean, R) = mul!(G, m.U', R)
_whiten_adjoint!(G, m::ProjectedMisfit, R) = mul!(G, m.U', R)
# Uᵀ as a dense n_obs × k matrix (Jacobians)
_whitening_adjoint_matrix(::SquaredEuclidean, n) = Matrix(1.0I, n, n)
_whitening_adjoint_matrix(m::WeightedSquaredEuclidean, n) = Matrix(m.U')
_whitening_adjoint_matrix(m::ProjectedMisfit, n) = Matrix{Float64}(m.U')

# X ← X - column means (voltages are only defined up to a constant)
function _remove_mean!(X)
    X .-= sum(X; dims = 1) ./ size(X, 1)
    return X
end

# Π X = X - 1 (wᵀX)/(wᵀ1): removes the w-weighted mean of every column (Π1 = 0). With w the
# boundary lengths of the measurements this is the discrete ∫_Γ-mean, consistent with the
# grounding ∫_Γ u ds = 0; with w = 1 it is the plain mean.
function _project!(X, w::AbstractVector)
    X .-= (w' * X) ./ sum(w)
    return X
end
# Πᵀ G = G - w (1ᵀG)/(wᵀ1)
function _project_adjoint!(G, w::AbstractVector)
    G .-= w .* (sum(G; dims = 1) ./ sum(w))
    return G
end
