# Projected block conjugate gradient method for symmetric positive semidefinite systems
#
#     A X = B,    A : ℝⁿ → ℝⁿ symmetric,  ker(A) = V,  A positive definite on V⊥,
#
# i.e. for SPD operators on the quotient space ℝⁿ/V. The typical case is the EIT
# stiffness matrix Lσ = ∫ σ ∇φᵢ⋅∇φⱼ dΩ with pure Neumann boundary conditions, whose
# null space are the constants. See the wiki articles "Projected Conjugate Gradient",
# "Block Conjugate Gradient" and "Grounding of the Potential".
#
# All O(n) work is done with generic array operations (sparse × dense products, BLAS-3,
# broadcasting), so the same code runs on the CPU and on the GPU (e.g. with
# `CuSparseMatrixCSR` / `CuMatrix`). Only s×s and k×s matrices are moved to the host.

using LinearAlgebra
using SparseArrays
import AlgebraicMultigrid

# ---------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------

# The first `r*c` entries of a column-major buffer, viewed as an `r × c` matrix (no copy).
# On the GPU this stays a dense device array, so it can be passed to CUBLAS/CUSPARSE.
_block(X::AbstractMatrix, r::Integer, c::Integer) =
    size(X) == (r, c) ? X : reshape(view(vec(X), 1:(r * c)), r, c)
# first `r` columns
_cols(X::AbstractMatrix, r::Integer) = _block(X, size(X, 1), r)

# device (or host) block → freshly allocated small host matrix, and back
_host(D::AbstractMatrix{T}) where {T} = copyto!(Matrix{T}(undef, size(D)), D)

"""
    _gram!(G, X, Y)

`G ← XᵀY` for tall-skinny blocks (`n × s` with `n ≫ s`). Generic fallback: `mul!`. Array
backends can specialise this; the CUDA extension replaces cuBLAS GEMM by one GEMV per column
for small Float64 blocks, where cuBLAS picks a very slow kernel.
"""
_gram!(G, X, Y) = mul!(G, X', Y)

_as_matrix(B::AbstractMatrix) = B
_as_matrix(b::AbstractVector) = reshape(b, :, 1)

# ---------------------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------------------

"""
    BlockCGWorkspace(A, B; nullspace = nothing, grounding = nothing)

Preallocated storage for [`pbcg!`](@ref) with `size(B, 2)` right-hand sides. All `n × s`
buffers are allocated with `similar(B, …)`, so passing device arrays gives a device workspace.

- `nullspace`: basis of `V = ker(A)` as an `n × k` matrix or a vector. Default: the constant
  vector (pure Neumann problem). It does not need to be orthonormal.
- `grounding`: linear functionals `W` (`n × k` or vector) fixing the component in `V`; the
  returned solution satisfies `Wᵀx = 0`. Default: `W = V`, i.e. the solution orthogonal to the
  null space (minimum Euclidean norm). For EIT with the boundary sum fixed to zero, pass the
  indicator vector of the boundary degrees of freedom (see [`boundary_grounding`](@ref)).
  `WᵀV` must be invertible.
"""
mutable struct BlockCGWorkspace{T, MT <: AbstractMatrix{T}}
    R::MT                 # residual                         n × s
    Z::MT                 # preconditioned residual / next P n × s
    P::MT                 # search directions                n × s
    Q::MT                 # A P (also scratch)               n × s
    V::MT                 # orthonormal null-space basis     n × k
    W::MT                 # grounding functionals            n × k
    F::MT                 # V (WᵀV)⁻¹                         n × k
    Ks::MT                # k × s device buffer
    Ss::MT                # s × s device buffer
    Ss2::MT               # s × s device buffer
    mask::MT              # 1 × s device buffer (deflation mask)
    maskh::Matrix{T}      # 1 × s host buffer
    nrm::MT               # 1 × s device buffer (column norms)
    nrmh::Vector{T}       # s host buffer
    dvec::Vector{T}       # s host buffers
    bnorm::Vector{T}
    rnorm::Vector{T}
    tol::Vector{T}
    active::Vector{Bool}
end

function BlockCGWorkspace(A, B::AbstractVecOrMat; nullspace = nothing, grounding = nothing)
    Bm = _as_matrix(B)
    n, s = size(Bm)
    size(A, 1) == size(A, 2) == n || throw(DimensionMismatch("A must be $n × $n"))
    T = eltype(Bm)

    Vh = nullspace === nothing ? ones(T, n, 1) : T.(Array(_as_matrix(nullspace)))
    size(Vh, 1) == n || throw(DimensionMismatch("null-space basis must have $n rows"))
    Vh = Matrix(qr(Vh).Q)                                   # orthonormal basis of V
    k = size(Vh, 2)
    Wh = grounding === nothing ? copy(Vh) : T.(Array(_as_matrix(grounding)))
    size(Wh) == (n, k) || throw(DimensionMismatch("grounding must be $n × $k"))
    WV = Wh' * Vh
    abs(det(WV)) > eps(T) * opnorm(Wh) || throw(ArgumentError("WᵀV must be invertible"))
    Fh = Vh / WV

    dev(M) = copyto!(similar(Bm, size(M)...), M)
    buf(r, c) = fill!(similar(Bm, r, c), zero(T))
    return BlockCGWorkspace{T, typeof(buf(1, 1))}(
        buf(n, s), buf(n, s), buf(n, s), buf(n, s),
        dev(Vh), dev(Wh), dev(Fh), buf(k, s), buf(s, s), buf(s, s), buf(1, s), ones(T, 1, s), buf(1, s), zeros(T, s),
        zeros(T, s), zeros(T, s), zeros(T, s), zeros(T, s), fill(true, s))
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

# ---------------------------------------------------------------------------------------
# Rank-revealing orthonormalisation of the search block (SVQB)
# ---------------------------------------------------------------------------------------

# Replaces the leading columns of P by an orthonormal basis of span(P) and returns its
# dimension r. Columns are first normalised, so only (near) linear dependence, not scale,
# decides which directions are dropped. Uses ws.Q as n × s scratch.
function _orthonormalize!(ws::BlockCGWorkspace{T}, s::Int, rank_tol) where {T}
    P, Q = ws.P, ws.Q
    G = _block(ws.Ss, s, s)
    _gram!(G, P, P)
    Gh = _host(G)
    d = ws.dvec
    tiny = floatmin(T) / eps(T)
    @inbounds for j in 1:s
        d[j] = Gh[j, j] > tiny ? inv(sqrt(Gh[j, j])) : zero(T)
    end
    @inbounds for j in 1:s, i in 1:s
        Gh[i, j] *= d[i] * d[j]
    end
    E = eigen!(Symmetric(Gh))                              # O(s³) on the host
    λmax = maximum(E.values; init = zero(T))
    λmax > 0 || return 0
    Th = zeros(T, s, s)
    r = 0
    @inbounds for j in s:-1:1                               # largest eigenvalues first
        λ = E.values[j]
        λ > rank_tol * λmax || continue
        r += 1
        c = inv(sqrt(λ))
        for i in 1:s
            Th[i, r] = d[i] * E.vectors[i, j] * c
        end
    end
    r == 0 && return 0
    Tr = _block(ws.Ss2, s, r)
    copyto!(Tr, Th[:, 1:r])
    Qr = _cols(Q, r)
    mul!(Qr, P, Tr)
    copyto!(_cols(P, r), Qr)
    return r
end

# Solve the small SPD system G Y = C on the host (Cholesky, eigen-based pseudo-inverse as
# fallback). G is r × r, C is r × s; the result overwrites C.
function _small_spd_solve!(G::AbstractMatrix{T}, C::AbstractMatrix{T}) where {T}
    Gs = Symmetric(G)
    F = cholesky!(copy(Gs); check = false)
    if issuccess(F)
        ldiv!(F, C)
    else
        E = eigen(Gs)
        λmax = maximum(abs, E.values)
        λinv = [λ > sqrt(eps(T)) * λmax ? inv(λ) : zero(T) for λ in E.values]
        C .= E.vectors * (Diagonal(λinv) * (E.vectors' * C))
    end
    return C
end

# ---------------------------------------------------------------------------------------
# Preconditioners
# ---------------------------------------------------------------------------------------

"""
    apply_preconditioner!(Z, M, R)

`Z ← M⁻¹ R` for a block of vectors. `M = nothing` is the identity.
"""
apply_preconditioner!(Z, ::Nothing, R) = copyto!(Z, R)

"""
    JacobiPreconditioner(A; to_device = identity)

Diagonal (Jacobi) preconditioner `M = diag(A)`. `A` must be a host matrix; `to_device`
converts the stored inverse diagonal (e.g. `CuArray`).
"""
struct JacobiPreconditioner{VT}
    dinv::VT
end
JacobiPreconditioner(A::AbstractMatrix; to_device = identity) =
    JacobiPreconditioner(to_device(inv.(Vector(diag(A)))))

apply_preconditioner!(Z, M::JacobiPreconditioner, R) = (Z .= M.dinv .* R; Z)

struct AMGLevel{T, MA, MR, VT, MT}
    A::MA                 # level operator
    P::MA                 # prolongation  (coarse → this level)
    R::MR                 # restriction   (= Pᵀ, stored explicitly)
    dinv::VT              # inverse diagonal for the Jacobi smoother
    ω::T                  # Jacobi damping
    x::MT                 # buffers for the coarse-level correction (size of next level)
    b::MT
    r::MT                 # residual buffer (size of this level)
end

"""
    AMGPreconditioner(A, s; to_device = identity, sweeps = 1, max_coarse = 64, max_levels = 10)

Symmetric smoothed-aggregation AMG V-cycle for `s` right-hand sides at once.

The hierarchy (aggregation, prolongations `P`, Galerkin coarse operators `PᵀAP`) is built on the
host with AlgebraicMultigrid.jl, using the constants as near-null space. The cycle itself only
uses sparse × dense products, damped Jacobi smoothing and a dense pseudo-inverse on the
coarsest level, so it runs unchanged on the GPU when `to_device` moves matrices there, e.g.

    to_device = x -> x isa SparseMatrixCSC ? CuSparseMatrixCSR(x) : CuArray(x)

With the same number of pre- and post-smoothing sweeps and `R = Pᵀ` the cycle is a symmetric
operator, positive definite on the complement of the constants, as required by CG. The
Jacobi damping is `ω = 4 / (3 ρ(D⁻¹A))`, with ρ estimated by power iteration.
"""
struct AMGPreconditioner{L, MC}
    levels::Vector{L}
    coarse_pinv::MC       # dense pseudo-inverse of the coarsest operator (handles ker = constants)
    sweeps::Int
end

function _jacobi_weight(A::SparseMatrixCSC{T}, dinv::Vector{T}; iters = 30) where {T}
    x = normalize!(rand(T, size(A, 1)) .- T(0.5))
    ρ = one(T)
    for _ in 1:iters
        y = dinv .* (A * x)
        ρ = norm(y)
        ρ == 0 && return one(T)
        x .= y ./ ρ
    end
    return T(4) / (3 * T(1.05) * ρ)                         # 5 % safety margin on ρ
end

function AMGPreconditioner(A::SparseMatrixCSC, s::Integer; to_device = identity, sweeps::Integer = 1,
                           max_coarse::Integer = 64, max_levels::Integer = 10)
    T = eltype(A)
    ml = AlgebraicMultigrid.smoothed_aggregation(A; max_coarse, max_levels,
                                                 coarse_solver = AlgebraicMultigrid.Pinv)
    dense(r, c) = to_device(zeros(T, r, c))
    levels = map(ml.levels) do lvl
        Al = SparseMatrixCSC{T, Int}(lvl.A)
        Pl = SparseMatrixCSC{T, Int}(lvl.P)
        Rl = SparseMatrixCSC{T, Int}(sparse(transpose(Pl)))
        dinv = inv.(Vector(diag(Al)))
        ω = _jacobi_weight(Al, dinv)
        nc = size(Pl, 2)
        AMGLevel(to_device(Al), to_device(Pl), to_device(Rl), to_device(dinv), ω,
                 dense(nc, s), dense(nc, s), dense(size(Al, 1), s))
    end
    Ac = Matrix(ml.final_A)
    return AMGPreconditioner(levels, to_device(T.(pinv(Ac))), Int(sweeps))
end

function _vcycle!(x, M::AMGPreconditioner, b, l::Int)
    if l > length(M.levels)
        mul!(x, M.coarse_pinv, b)
        return x
    end
    L = M.levels[l]
    r = _cols(L.r, size(b, 2))
    ω, dinv = L.ω, L.dinv
    @. x = ω * dinv * b                                    # first sweep from x = 0
    for _ in 2:M.sweeps
        mul!(r, L.A, x)
        @. x += ω * dinv * (b - r)
    end
    mul!(r, L.A, x)
    @. r = b - r
    xc, bc = _cols(L.x, size(b, 2)), _cols(L.b, size(b, 2))
    mul!(bc, L.R, r)
    _vcycle!(xc, M, bc, l + 1)
    mul!(x, L.P, xc, 1, 1)
    for _ in 1:M.sweeps
        mul!(r, L.A, x)
        @. x += ω * dinv * (b - r)
    end
    return x
end

apply_preconditioner!(Z, M::AMGPreconditioner, R) = _vcycle!(Z, M, R, 1)

# ---------------------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------------------

"""
Result information of [`pbcg!`](@ref).

- `converged`: all columns reached their tolerance
- `iterations`: number of block iterations
- `residuals`: final relative residuals `‖Π(bⱼ - A xⱼ)‖ / ‖Π bⱼ‖` per column
- `compatibility_defect`: `‖(I - Π) B‖ / ‖B‖`, the part of the right-hand side outside
  `range(A) = V⊥` that was removed (should be ≈ 0 for a consistent Neumann problem)
"""
struct BlockCGStats{T}
    converged::Bool
    iterations::Int
    residuals::Vector{T}
    compatibility_defect::T
end

"""
    pbcg!(X, ws, A, B; M = nothing, rtol = √eps, atol = 0, maxiter = 10n, recompute_every = 50,
          rank_tol = √eps)

Solve `A X = Π B` with the projected block conjugate gradient method, in place in `X` (which
is used as initial guess), and return a [`BlockCGStats`](@ref).

`Π = I - V Vᵀ` is the orthogonal projector onto `V⊥ = range(A)`. Every residual and every
preconditioned residual is projected, so the iteration never leaves `V⊥` (round-off and
preconditioners that do not preserve `V⊥` cannot pollute the solution, and inconsistent data
are handled by solving the projected, consistent problem). At the end the component in `V`
is fixed by the grounding condition `Wᵀx = 0` of the workspace.

Block iteration (O'Leary 1980, in Galerkin form with orthonormalised search blocks):

    P ← orthonormal basis of span(P)      (rank-revealing; handles dependent columns)
    α = (PᵀAP)⁻¹ PᵀR,   X += Pα,   R -= APα
    Z = Π M⁻¹ R,        β = -(PᵀAP)⁻¹ (AP)ᵀZ,   P ← Z + Pβ

Columns that have converged are removed from the block (deflation). `recompute_every`
replaces the updated residual by the true residual `Π(B - AX)` to limit round-off drift.
"""
function pbcg!(X::AbstractVecOrMat, ws::BlockCGWorkspace{T}, A, B::AbstractVecOrMat;
               M = nothing, rtol = sqrt(eps(T)), atol = zero(T),
               maxiter::Integer = 10 * size(A, 1), recompute_every::Integer = 50,
               rank_tol = sqrt(eps(T))) where {T}
    Xm, Bm = _as_matrix(X), _as_matrix(B)
    s = size(Bm, 2)
    size(ws.R, 2) == s || throw(DimensionMismatch("workspace was built for $(size(ws.R, 2)) right-hand sides"))
    R, bnorm, rnorm, tol, active = ws.R, ws.bnorm, ws.rnorm, ws.tol, ws.active

    # compatibility: ‖B‖² = ‖ΠB‖² + ‖(I-Π)B‖²
    _colnorms!(rnorm, Bm, ws)
    nB² = sum(abs2, rnorm)
    copyto!(R, Bm)
    _project!(R, ws)
    _colnorms!(bnorm, R, ws)
    defect = nB² > 0 ? sqrt(max(nB² - sum(abs2, bnorm), zero(T)) / nB²) : zero(T)
    @inbounds for j in 1:s
        tol[j] = max(T(atol), T(rtol) * bnorm[j])
    end

    # initial residual R = Π(B - A X) with X ∈ V⊥
    _project!(Xm, ws)
    mul!(R, A, Xm)
    @. R = Bm - R
    _project!(R, ws)
    _colnorms!(rnorm, R, ws)
    @inbounds for j in 1:s
        active[j] = rnorm[j] > tol[j]
    end

    iter = 0
    converged = !any(active)
    if !converged
        _precondition!(ws, M)
        copyto!(ws.P, ws.Z)
    end
    while !converged && iter < maxiter
        iter += 1
        r = _orthonormalize!(ws, s, rank_tol)
        r == 0 && break                                     # no search direction left
        Pr, Qr = _cols(ws.P, r), _cols(ws.Q, r)
        mul!(Qr, A, Pr)

        # G = PᵀAP (r × r) and α = G⁻¹ PᵀR (r × s), both solved on the host
        Gd = _block(ws.Ss, r, r)
        _gram!(Gd, Pr, Qr)
        Gh = _host(Gd)
        @inbounds for j in 1:r, i in 1:j                    # symmetrise
            Gh[i, j] = Gh[j, i] = (Gh[i, j] + Gh[j, i]) / 2
        end
        Cdr = _block(ws.Ss2, r, s)
        _gram!(Cdr, Pr, R)
        Ch = _host(Cdr)
        _small_spd_solve!(copy(Gh), Ch)                     # Ch ← α
        copyto!(Cdr, Ch)

        mul!(Xm, Pr, Cdr, 1, 1)
        if iter % recompute_every == 0
            mul!(R, A, Xm)
            @. R = Bm - R
            _project!(R, ws)
        else
            mul!(R, Qr, Cdr, -1, 1)
        end

        _colnorms!(rnorm, R, ws)
        @inbounds for j in 1:s
            active[j] = rnorm[j] > tol[j]
        end
        converged = !any(active)
        converged && break

        # next search block: P ← Z + Pβ with β = -G⁻¹ QᵀZ (A-conjugate to the current P)
        _precondition!(ws, M)
        _gram!(Cdr, Qr, ws.Z)
        copyto!(Ch, Cdr)
        _small_spd_solve!(copy(Gh), Ch)
        Ch .*= -1
        copyto!(Cdr, Ch)
        mul!(ws.Z, Pr, Cdr, 1, 1)
        ws.P, ws.Z = ws.Z, ws.P
    end

    _ground!(Xm, ws)
    return BlockCGStats(converged, iter, rnorm ./ max.(bnorm, floatmin(T)), defect)
end

# Z ← Π M⁻¹ R, with converged columns zeroed (deflation)
function _precondition!(ws::BlockCGWorkspace{T}, M) where {T}
    apply_preconditioner!(ws.Z, M, ws.R)
    _project!(ws.Z, ws)
    s = size(ws.Z, 2)
    if !all(ws.active)
        @inbounds for j in 1:s
            ws.maskh[1, j] = ws.active[j] ? one(T) : zero(T)
        end
        copyto!(ws.mask, ws.maskh)
        ws.Z .*= ws.mask
    end
    return ws.Z
end

"""
    X, stats = pbcg(A, B; nullspace = nothing, grounding = nothing, M = nothing, kwargs...)

Allocating convenience wrapper around [`BlockCGWorkspace`](@ref) and [`pbcg!`](@ref). `B` may
be a vector or an `n × s` matrix (on the CPU or the GPU).
"""
function pbcg(A, B::AbstractVecOrMat; nullspace = nothing, grounding = nothing, kwargs...)
    X = fill!(similar(B), zero(eltype(B)))
    ws = BlockCGWorkspace(A, B; nullspace, grounding)
    stats = pbcg!(X, ws, A, B; kwargs...)
    return X, stats
end
