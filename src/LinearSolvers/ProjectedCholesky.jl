# Projected sparse Cholesky solver for symmetric positive semidefinite systems
#
#     A X = B,    ker(A) = V,  A positive definite on V⊥   (A SPD on the quotient ℝⁿ/V),
#
# e.g. the EIT stiffness matrix Lσ with pure Neumann boundary conditions (V = constants).
#
# Idea: choose k = dim V "pinned" degrees of freedom I such that V[I, :] is invertible. Then the
# principal submatrix A[J, J] on the complement J is symmetric positive definite and as sparse as
# A, so it has an ordinary sparse Cholesky factorisation. The solution with x[I] = 0 solves the
# (projected, consistent) system, and the grounding condition Wᵀx = 0 is imposed afterwards by an
# oblique projection along V. See the wiki articles "Projected Cholesky Factorization" and
# "Grounding of the Potential".
#
# Backends: CHOLMOD (CPU, Float64) and, through the package extension ModularEITCUDSSExt,
# NVIDIA cuDSS on the GPU (Float32/Float64). All O(n) work outside the factorisation uses generic
# array operations, as in the projected block CG solver.

using LinearAlgebra
using SparseArrays
import LDLFactorizations

# ---------------------------------------------------------------------------------------
# Backend interface (CPU: CHOLMOD). Device backends add methods for their matrix types.
# ---------------------------------------------------------------------------------------

# factorise the reduced SPD matrix
_chol_factorize(A::SparseMatrixCSC{Float64}) = cholesky(Hermitian(A, :L))
_chol_factorize(::SparseMatrixCSC{T}) where {T} =
    throw(ArgumentError("the CPU backend (CHOLMOD) supports Float64 only, got $T"))

# overwrite the values of the (device) reduced matrix; the pattern is unchanged
_set_values!(A::SparseMatrixCSC, nzval::Vector) = (copyto!(A.nzval, nzval); A)

# numeric refactorisation with the symbolic analysis of the first factorisation
_chol_refactor!(F::SparseArrays.CHOLMOD.Factor, A::SparseMatrixCSC) = (cholesky!(F, Hermitian(A, :L)); F)

# X ← A⁻¹ B for the reduced system (X, B are nJ × s)
_chol_solve!(X, F::SparseArrays.CHOLMOD.Factor, B) = ldiv!(X, F, B)

# LDLᵀ backend (LDLFactorizations.jl, pure Julia, any floating-point type). It reads the upper
# triangle only, so the reduced matrix is stored as triu(A[J, J]) and marked with this wrapper.
struct _UpperTriangle{M}
    A::M
end
_chol_factorize(U::_UpperTriangle) = LDLFactorizations.ldl(Symmetric(U.A, :U))
_set_values!(U::_UpperTriangle, nzval::Vector) = (copyto!(U.A.nzval, nzval); U)
_chol_refactor!(F::LDLFactorizations.LDLFactorization, U::_UpperTriangle) =
    (LDLFactorizations.ldl_factorize!(Symmetric(U.A, :U), F); F)
_chol_solve!(X, F::LDLFactorizations.LDLFactorization, B) = ldiv!(X, F, B)

# ---------------------------------------------------------------------------------------
# Solver object
# ---------------------------------------------------------------------------------------

"""
    projected_cholesky(A; nullspace = nothing, grounding = nothing, nrhs = 1, to_device = identity)

Sparse Cholesky solver for `A X = B` where `A` (a `SparseMatrixCSC`) is symmetric positive
semidefinite with null space `V` and positive definite on `V⊥`, i.e. an SPD operator on `ℝⁿ/V`.
Returns a [`ProjectedCholesky`](@ref) that solves with `F \\ B` or `ldiv!(X, F, B)` for a vector or
a block of right-hand sides.

- `nullspace`: basis of `V` (`n × k` matrix or vector; default: constants, the pure Neumann case).
- `grounding`: functionals `W` fixing the component in `V`, the solution satisfies `Wᵀx = 0`
  (default `W = V`: solution orthogonal to `V`). For EIT, `boundary_grounding(n, boundary_dofs)`
  makes the boundary values sum to zero; see [`boundary_grounding`](@ref).
- `nrhs`: number of right-hand sides to preallocate buffers for (other sizes reallocate once).
- `to_device`: converts matrices/vectors to a device. With CUDA.jl and CUDSS.jl loaded,
  `to_device = x -> x isa SparseMatrixCSC ? CuSparseMatrixCSR(x) : CuArray(x)` factorises and
  solves on the GPU with cuDSS.
- `backend`: `:cholmod` (CPU, Float64, supernodal), `:ldl` (CPU, LDLFactorizations.jl, any
  floating-point type) or `:auto` (default: CHOLMOD for Float64 on the CPU, LDLᵀ for other types,
  the device backend when `to_device` is given). See also [`projected_ldl`](@ref).

Right-hand sides are projected onto `range(A) = V⊥` first, so inconsistent data are solved in the
least-squares sense. Use [`refactor!`](@ref) after the matrix values change (same pattern).
"""
function projected_cholesky(A::SparseMatrixCSC; nullspace = nothing, grounding = nothing,
                            nrhs::Integer = 1, to_device = identity, backend::Symbol = :auto)
    return ProjectedCholesky(A; nullspace, grounding, nrhs, to_device, backend)
end

"""
    projected_ldl(A; kwargs...)

[`projected_cholesky`](@ref) with the LDLᵀ backend of LDLFactorizations.jl (`backend = :ldl`):
pure Julia, works for `Float32`, `Float64` and other floating-point types, CPU only.
"""
projected_ldl(A::SparseMatrixCSC; kwargs...) = projected_cholesky(A; kwargs..., backend = :ldl)

"""
    ProjectedCholesky

Factorisation object returned by [`projected_cholesky`](@ref). Solve with `F \\ B` or
`ldiv!(X, F, B)`; update the numerical values with [`refactor!`](@ref).
"""
mutable struct ProjectedCholesky{T, MA, FT, MT <: AbstractMatrix{T}, IT <: AbstractVector{<:Integer}, DV}
    n::Int
    pinned::Vector{Int}            # I: pinned degrees of freedom (x[I] = 0 before grounding)
    J::IT                          # complement of I (on the device)
    Ared::SparseMatrixCSC{T, Int}  # host copy of A[J, J]
    nzmap::Vector{Int}             # Ared.nzval = A.nzval[nzmap]
    Adev::MA                       # A[J, J] on the device (== Ared on the CPU)
    fact::FT                       # backend factorisation of A[J, J]
    V::MT                          # orthonormal null-space basis   n × k
    VJ::MT                         # V[J, :]                       nJ × k
    W::MT                          # grounding functionals          n × k
    F::MT                          # V (WᵀV)⁻¹                      n × k
    K::MT                          # k × s buffer
    bJ::MT                         # nJ × s buffers
    xJ::MT
    to_device::DV
end

# choose k pinned indices with V[I, :] well conditioned (column-pivoted QR of Vᵀ)
_pinned_dofs(V::Matrix) = sort(qr(copy(V'), ColumnNorm()).p[1:size(V, 2)])

function ProjectedCholesky(A::SparseMatrixCSC{T}; nullspace = nothing, grounding = nothing,
                           nrhs::Integer = 1, to_device = identity, backend::Symbol = :auto) where {T}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("A must be square"))
    Vh = nullspace === nothing ? ones(T, n, 1) : T.(Array(_as_matrix(nullspace)))
    size(Vh, 1) == n || throw(DimensionMismatch("null-space basis must have $n rows"))
    Vh = Matrix(qr(Vh).Q)
    k = size(Vh, 2)
    Wh = grounding === nothing ? copy(Vh) : T.(Array(_as_matrix(grounding)))
    size(Wh) == (n, k) || throw(DimensionMismatch("grounding must be $n × $k"))
    WV = Wh' * Vh
    abs(det(WV)) > eps(T) * opnorm(Wh) || throw(ArgumentError("WᵀV must be invertible"))

    pinned = _pinned_dofs(Vh)
    J = setdiff(1:n, pinned)
    # value map A.nzval → A[J, J].nzval, so refactorisation needs no sparse indexing
    if backend == :auto
        backend = to_device !== identity ? :device : T == Float64 ? :cholmod : :ldl
    end
    backend in (:cholmod, :ldl, :device) || throw(ArgumentError("unknown backend $backend"))
    backend == :ldl && to_device !== identity &&
        throw(ArgumentError("the LDLᵀ backend runs on the CPU only"))
    Aidx = SparseMatrixCSC(n, n, A.colptr, A.rowval, collect(1.0:length(A.nzval)))
    AidxJJ = backend == :ldl ? triu(Aidx[J, J]) : Aidx[J, J]
    nzmap = round.(Int, AidxJJ.nzval)
    Ared = SparseMatrixCSC{T, Int}(AidxJJ.m, AidxJJ.n, AidxJJ.colptr, AidxJJ.rowval, A.nzval[nzmap])

    Adev = backend == :ldl ? _UpperTriangle(Ared) : to_device(Ared)
    fact = _chol_factorize(Adev)
    nJ = length(J)
    dev(M) = to_device(M)
    buf(r, c) = to_device(zeros(T, r, c))
    return ProjectedCholesky(n, pinned, to_device(J), Ared, nzmap, Adev, fact,
                             dev(Vh), dev(Vh[J, :]), dev(Wh), dev(Vh / WV),
                             buf(k, nrhs), buf(nJ, nrhs), buf(nJ, nrhs), to_device)
end

Base.size(F::ProjectedCholesky) = (F.n, F.n)
Base.size(F::ProjectedCholesky, i::Integer) = i <= 2 ? F.n : 1
Base.eltype(::ProjectedCholesky{T}) where {T} = T

"""
    refactor!(F::ProjectedCholesky, A)

Numerical refactorisation for new values of `A` with the same sparsity pattern (e.g. a new
conductivity σ in the same mesh). Reuses the symbolic analysis (ordering, elimination tree).
"""
function refactor!(F::ProjectedCholesky, A::SparseMatrixCSC)
    size(A) == (F.n, F.n) || throw(DimensionMismatch("matrix size changed"))
    Ared = F.Ared
    @inbounds for (i, p) in enumerate(F.nzmap)
        Ared.nzval[i] = A.nzval[p]
    end
    _set_values!(F.Adev, Ared.nzval)
    F.fact = _chol_refactor!(F.fact, F.Adev)
    return F
end

function _ensure_buffers!(F::ProjectedCholesky{T}, s::Integer) where {T}
    if size(F.bJ, 2) != s
        nJ, k = length(F.J), size(F.V, 2)
        F.K = F.to_device(zeros(T, k, s))
        F.bJ = F.to_device(zeros(T, nJ, s))
        F.xJ = F.to_device(zeros(T, nJ, s))
    end
    return F
end

"""
    ldiv!(X, F::ProjectedCholesky, B)

Solve `A X = Π B` and ground the solution (`Wᵀ X = 0`). `X` and `B` are vectors or `n × s`
matrices (on the device of `F`).
"""
function LinearAlgebra.ldiv!(X::AbstractVecOrMat, F::ProjectedCholesky, B::AbstractVecOrMat)
    Xm, Bm = _as_matrix(X), _as_matrix(B)
    s = size(Bm, 2)
    _ensure_buffers!(F, s)
    K, bJ, xJ = F.K, F.bJ, F.xJ
    # b_J = (Π B)[J, :] = B[J, :] - V[J, :] (Vᵀ B)
    _gram!(K, F.V, Bm)
    bJ .= view(Bm, F.J, :)
    mul!(bJ, F.VJ, K, -1, 1)
    _chol_solve!(xJ, F.fact, bJ)
    # X = [x_J; 0] then ground: X ← X - V (WᵀV)⁻¹ Wᵀ X
    fill!(Xm, zero(eltype(Xm)))
    view(Xm, F.J, :) .= xJ
    _gram!(K, F.W, Xm)
    mul!(Xm, F.F, K, -1, 1)
    return X
end

Base.:\(F::ProjectedCholesky, B::AbstractVecOrMat) = ldiv!(similar(B), F, B)
