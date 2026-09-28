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

# r ← U e
_whiten!(R, ::SquaredEuclidean, E) = copyto!(R, E)
_whiten!(R, m::WeightedSquaredEuclidean, E) = mul!(R, m.U, E)
# G ← Uᵀ R
_whiten_adjoint!(G, ::SquaredEuclidean, R) = copyto!(G, R)
_whiten_adjoint!(G, m::WeightedSquaredEuclidean, R) = mul!(G, m.U', R)
# Uᵀ as a dense n_obs × n_obs matrix (Jacobians)
_whitening_adjoint_matrix(::SquaredEuclidean, n) = Matrix(1.0I, n, n)
_whitening_adjoint_matrix(m::WeightedSquaredEuclidean, n) = Matrix(m.U')

# X ← X - column means (voltages are only defined up to a constant)
function _remove_mean!(X)
    X .-= sum(X; dims = 1) ./ size(X, 1)
    return X
end
