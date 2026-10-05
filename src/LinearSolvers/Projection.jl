# Projection onto the complement of a known null space and grounding, shared by the projected
# solvers (block CG via Krylov.jl, block MINRES, the projected Cholesky factorisation):
#
#     A X = B,    A : ℝⁿ → ℝⁿ symmetric,  ker(A) = V,  A positive definite on V⊥.
#
# See the wiki articles "Projected Conjugate Gradient" and "Grounding of the Potential".

using LinearAlgebra
import Krylov

# ---------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------

# The first `r*c` entries of a column-major buffer, viewed as an `r × c` matrix (no copy).
# On the GPU this stays a dense device array, so it can be passed to CUBLAS/CUSPARSE.
_block(X::AbstractMatrix, r::Integer, c::Integer) =
    size(X) == (r, c) ? X : reshape(view(vec(X), 1:(r * c)), r, c)
# first `r` columns
_cols(X::AbstractMatrix, r::Integer) = _block(X, size(X, 1), r)


"""
    _gram!(G, X, Y)

`G ← XᵀY` for tall-skinny blocks (`n × s` with `n ≫ s`): `Krylov.kgram!` (one GEMV per column for
double-precision blocks with 2–8 columns, where GEMM kernels are slow).
"""
_gram!(G, X, Y) = Krylov.kgram!(G, X, Y)

_as_matrix(B::AbstractMatrix) = B
_as_matrix(b::AbstractVector) = reshape(b, :, 1)

# Orthonormal null-space basis V, grounding functionals W and F = V (WᵀV)⁻¹ on the host, from the
# `nullspace` (default: the constants) and `grounding` (default: W = V) arguments of the solvers.
function _nullspace_basis(T, n, nullspace, grounding)
    Vh = nullspace === nothing ? ones(T, n, 1) : T.(Array(_as_matrix(nullspace)))
    size(Vh, 1) == n || throw(DimensionMismatch("null-space basis must have $n rows"))
    Vh = Matrix(qr(Vh).Q)                                   # orthonormal basis of V
    k = size(Vh, 2)
    Wh = grounding === nothing ? copy(Vh) : T.(Array(_as_matrix(grounding)))
    size(Wh) == (n, k) || throw(DimensionMismatch("grounding must be $n × $k"))
    WV = Wh' * Vh
    abs(det(WV)) > eps(T) * opnorm(Wh) || throw(ArgumentError("WᵀV must be invertible"))
    return Vh, Wh, Vh / WV
end

"""
    boundary_grounding(n, dofs; weights = nothing)

Grounding functional `w` with `wᵢ = 1` (or `weights`) on the degrees of freedom `dofs`, so that
`wᵀx = Σ_{i ∈ dofs} xᵢ = 0`. Pass it as `grounding` to [`pbcg`](@ref).
"""
function boundary_grounding(n::Integer, dofs; weights = nothing, T = Float64)
    w = zeros(T, n)
    w[dofs] .= weights === nothing ? one(T) : weights
    return w
end

# X ← (I - V Vᵀ) X : orthogonal projection onto V⊥ (V orthonormal)
function _project!(X, ws)  # ws: any object with fields V, Ks
    r = size(X, 2)
    K = _cols(ws.Ks, r)
    _gram!(K, ws.V, X)
    mul!(X, ws.V, K, -1, 1)
    return X
end

# X ← X - V (WᵀV)⁻¹ Wᵀ X : oblique projection onto {Wᵀx = 0} along V
function _ground!(X, ws)   # ws: any object with fields W, F, Ks
    K = _cols(ws.Ks, size(X, 2))
    _gram!(K, ws.W, X)
    mul!(X, ws.F, K, -1, 1)
    return X
end

# Euclidean column norms of X: one O(n s) reduction on the device, s values to the host.
function _colnorms!(out::Vector, X, ws)  # ws: fields nrm, nrmh
    s = size(X, 2)
    buf = _block(ws.nrm, 1, s)
    sum!(abs2, buf, X)
    copyto!(ws.nrmh, 1, buf, 1, s)
    @inbounds for j in 1:s
        out[j] = sqrt(max(ws.nrmh[j], zero(eltype(out))))
    end
    return out
end
