# Projected block conjugate gradients for symmetric positive semidefinite systems
#
#     A X = B,    A : ℝⁿ → ℝⁿ symmetric,  ker(A) = V,  A positive definite on V⊥,
#
# i.e. for SPD operators on the quotient space ℝⁿ/V. The typical case is the EIT stiffness matrix
# Lσ = ∫ σ ∇φᵢ⋅∇φⱼ dΩ with pure Neumann boundary conditions, whose null space are the constants.
# The solver itself is `block_cg` of the Krylov.jl fork (github.com/DanielBoigk/Krylov.jl,
# branch block-cg); this file adds the EIT defaults (null space = constants), the preconditioners
# (Jacobi, smoothed-aggregation AMG), and the interface used by the rest of the package. See the
# wiki articles "Projected Conjugate Gradient", "Block Conjugate Gradient" and "Grounding of the
# Potential".
#
# All O(n) work is done with generic array operations, so the same code runs on the CPU and on
# any GPUArrays backend (sparse matrices as DeviceSparseMatrixCSR or vendor types).

using LinearAlgebra
using SparseArrays
import AlgebraicMultigrid
import Krylov

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
converts the stored inverse diagonal (e.g. `device_converter(CuArray)`).
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
coarsest level, so it runs unchanged on any GPU backend when `to_device` moves matrices there,
e.g. `to_device = device_converter(CuArray)` (or `ROCArray`, `oneArray`, `MtlArray`); see
[`device_converter`](@ref).

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

# A ModularEIT preconditioner as a Krylov.jl operator: `mul!(Z, op, R)` applies M⁻¹.
struct _KrylovPreconditioner{P}
    M::P
end
LinearAlgebra.mul!(Z, op::_KrylovPreconditioner, R) = apply_preconditioner!(Z, op.M, R)
_krylov_preconditioner(::Nothing) = I
_krylov_preconditioner(M) = _KrylovPreconditioner(M)

# ---------------------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------------------

"""
    BlockCGWorkspace(A, B; nullspace = nothing, grounding = nothing)

Preallocated storage for [`pbcg!`](@ref) with `size(B, 2)` right-hand sides: a
`Krylov.BlockCgWorkspace` of the Krylov.jl fork. All `n × s` buffers have the array type of `B`,
so passing device arrays gives a device workspace.

- `nullspace`: basis of `V = ker(A)` as an `n × k` matrix or a vector. Default: the constant
  vector (pure Neumann problem). It does not need to be orthonormal. Pass an `n × 0` matrix for
  a positive definite `A` (e.g. a Dirichlet system).
- `grounding`: linear functionals `W` (`n × k` or vector) fixing the component in `V`; the
  returned solution satisfies `Wᵀx = 0`. Default: `W = V`, i.e. the solution orthogonal to the
  null space (minimum Euclidean norm). For EIT with the boundary sum fixed to zero, pass the
  indicator vector of the boundary degrees of freedom (see [`boundary_grounding`](@ref)).
  `WᵀV` must be invertible.
"""
function BlockCGWorkspace(A, B::AbstractVecOrMat; nullspace = nothing, grounding = nothing)
    Bm = _as_matrix(B)
    n = size(Bm, 1)
    size(A, 1) == size(A, 2) == n || throw(DimensionMismatch("A must be $n × $n"))
    T = eltype(Bm)
    V, W, _ = _nullspace_basis(T, n, nullspace, grounding)
    k = size(V, 2)
    return Krylov.BlockCgWorkspace(A, Bm; nullspace = k == 0 ? nothing : V, grounding = k == 0 ? nothing : W)
end

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

Solve `A X = Π B` with the projected block conjugate gradient method (`Krylov.block_cg!` of
the Krylov.jl fork) in the workspace `ws` (see [`BlockCGWorkspace`](@ref)), in place in `X`
(which is used as initial guess), and return a [`BlockCGStats`](@ref).

`Π = I - V Vᵀ` is the orthogonal projector onto `V⊥ = range(A)`. Every residual and every
preconditioned residual is projected, so the iteration never leaves `V⊥` (round-off and
preconditioners that do not preserve `V⊥` cannot pollute the solution, and inconsistent data
are handled by solving the projected, consistent problem). At the end the component in `V`
is fixed by the grounding condition `Wᵀx = 0` of the workspace.

`M` is a preconditioner applied by [`ModularEIT.apply_preconditioner!`](@ref)
([`JacobiPreconditioner`](@ref), [`AMGPreconditioner`](@ref), [`DCTPreconditioner`](@ref),
[`PolarPreconditioner`](@ref)) or `nothing`. Column `j` has converged when its residual is
below `atol + rtol ‖Π bⱼ‖`; converged columns are removed from the block (deflation), dependent
search directions are dropped (relative eigenvalue threshold `rank_tol`), and every
`recompute_every` iterations the updated residual is replaced by the true residual
`Π(B - AX)` to limit round-off drift.
"""
function pbcg!(X::AbstractVecOrMat, ws::Krylov.BlockCgWorkspace{T}, A, B::AbstractVecOrMat;
               M = nothing, rtol = sqrt(eps(T)), atol = zero(T),
               maxiter::Integer = 10 * size(A, 1), recompute_every::Integer = 50,
               rank_tol = sqrt(eps(T))) where {T}
    Xm, Bm = _as_matrix(X), _as_matrix(B)
    size(Bm, 2) == ws.p || throw(DimensionMismatch("workspace was built for $(ws.p) right-hand sides"))
    Krylov.block_cg!(ws, A, Bm, Xm; M = _krylov_preconditioner(M), atol = T(atol), rtol = T(rtol),
                     itmax = Int(maxiter), recompute_every = Int(recompute_every), rank_tol = T(rank_tol))
    copyto!(Xm, ws.X)
    return BlockCGStats(ws.stats.solved, ws.stats.niter, ws.rnorm ./ max.(ws.bnorm, floatmin(T)),
                        ws.defect)
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
