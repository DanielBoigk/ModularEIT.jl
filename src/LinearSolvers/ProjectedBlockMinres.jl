# Projected block MINRES (Krylov.jl) for symmetric positive semidefinite systems A X = B with
# known null space V, e.g. the EIT Neumann stiffness matrix (V = constants).
#
# The right-hand side is projected onto range(A) = V⊥ (consistent system), block MINRES from
# Krylov.jl solves it, and the grounding condition Wᵀx = 0 (e.g. zero sum over the boundary
# nodes) is imposed afterwards. MINRES only needs symmetry, so it is robust on the semidefinite
# system; started from zero on a consistent system it converges to a solution. Krylov.jl's
# block MINRES does not accept a preconditioner yet, so optional Jacobi preconditioning is done
# by the symmetric scaling D^{-1/2} A D^{-1/2} (see the wiki article "MINRES").

using LinearAlgebra
import Krylov

# Symmetrically scaled operator  S = D A D  (D diagonal), applied to blocks.
struct _ScaledOperator{T, MA, VT, MT}
    A::MA
    d::VT                 # diagonal of D
    buf::MT               # n × s buffer
end
Base.size(S::_ScaledOperator) = size(S.A)
Base.size(S::_ScaledOperator, i::Integer) = size(S.A, i)
Base.eltype(::_ScaledOperator{T}) where {T} = T
LinearAlgebra.issymmetric(::_ScaledOperator) = true
LinearAlgebra.ishermitian(::_ScaledOperator) = true
function LinearAlgebra.mul!(Y::AbstractVecOrMat, S::_ScaledOperator, X::AbstractVecOrMat)
    # contiguous piece of the buffer with the shape of X (a device array on the GPU)
    buf = reshape(view(vec(S.buf), 1:length(X)), size(X))
    buf .= S.d .* X
    mul!(Y, S.A, buf)
    Y .*= S.d
    return Y
end

"""
    ProjectedMinresWorkspace(A, B; nullspace = nothing, grounding = nothing,
                             scaling = :jacobi, diagonal = nothing)

Preallocated storage for [`pbminres!`](@ref) with `size(B, 2)` right-hand sides (Krylov.jl
block-MINRES workspace plus projection/grounding buffers, all created with `similar(B, …)`).
`nullspace` and `grounding` work as in [`BlockCGWorkspace`](@ref). `scaling = :jacobi` solves the
symmetrically scaled system `D^{-1/2} A D^{-1/2}` with `D = diag(A)`; pass `diagonal` if `diag(A)`
is not available for the matrix type (e.g. on the GPU).

`columnwise = true` solves the right-hand sides one after another with Krylov.jl's single-vector
MINRES instead of block MINRES. Use it on the GPU: Krylov.jl's block MINRES currently fails there
because of a method ambiguity between cuBLAS.jl and GPUArrays.jl in triangular `rdiv!`.
"""
struct ProjectedMinresWorkspace{T, MT <: AbstractMatrix{T}, KW, VT}
    kw::KW                # Krylov.BlockMinresWorkspace, or Krylov.MinresWorkspace (columnwise)
    Bhat::MT              # projected (and scaled) right-hand side, n × s
    buf::MT               # buffer for the scaled operator, n × s
    d::VT                 # D^{-1/2} (or nothing without scaling)
    V::MT                 # orthonormal null-space basis n × k
    W::MT                 # grounding functionals        n × k
    F::MT                 # V (WᵀV)⁻¹                     n × k
    Ks::MT                # k × s buffer
    nrm::MT               # 1 × s buffer (column norms)
    nrmh::Vector{T}
    bnorm::Vector{T}
    rnorm::Vector{T}
end

function ProjectedMinresWorkspace(A, B::AbstractVecOrMat; nullspace = nothing, grounding = nothing,
                                  scaling::Symbol = :jacobi, diagonal = nothing,
                                  columnwise::Bool = false)
    scaling in (:none, :jacobi) || throw(ArgumentError("scaling must be :none or :jacobi"))
    Bm = _as_matrix(B)
    n, s = size(Bm)
    T = eltype(Bm)
    size(A, 1) == size(A, 2) == n || throw(DimensionMismatch("A must be $n × $n"))
    Vh, Wh, Fh = _nullspace_basis(T, n, nullspace, grounding)
    dev(M) = copyto!(similar(Bm, size(M)...), M)
    buf(r, c) = fill!(similar(Bm, r, c), zero(T))
    d = nothing
    if scaling == :jacobi
        dh = diagonal === nothing ? Vector(diag(A)) : Vector(diagonal)
        all(>(0), dh) || throw(ArgumentError("Jacobi scaling needs a positive diagonal"))
        d = copyto!(similar(Bm, n), T.(inv.(sqrt.(dh))))
    end
    SV = typeof(similar(Bm, n))
    kw = columnwise ? Krylov.MinresWorkspace(n, n, SV) : Krylov.BlockMinresWorkspace(n, n, s, SV, typeof(Bm))
    return ProjectedMinresWorkspace(kw, similar(Bm), similar(Bm), d, dev(Vh), dev(Wh), dev(Fh),
                                    buf(size(Vh, 2), s), buf(1, s), zeros(T, s), zeros(T, s), zeros(T, s))
end

"""
Result information of [`pbminres!`](@ref): `converged`, `iterations`, Krylov.jl `status`
string and `compatibility_defect = ‖(I - Π) B‖ / ‖B‖`.
"""
struct BlockMinresStats{T}
    converged::Bool
    iterations::Int
    status::String
    compatibility_defect::T
end

"""
    stats = pbminres!(X, ws, A, B; atol = 0, rtol = √eps, itmax = 0)

Solve `A X = Π B` with Krylov.jl's block MINRES in the workspace `ws` (see
[`ProjectedMinresWorkspace`](@ref)) and ground the result (`Wᵀ X = 0`). `X`, `B` are vectors or
`n × s` matrices. Tolerances refer to the Frobenius norm of the (scaled) block residual, as in
Krylov.jl. `itmax = 0` uses Krylov.jl's default `2n/s`.

Right-hand sides whose columns are linearly dependent (or zero) are handled by solving for an
orthonormal basis of their span and recombining.
"""
function pbminres!(X::AbstractVecOrMat, ws::ProjectedMinresWorkspace{T}, A, B::AbstractVecOrMat;
                   atol = zero(T), rtol = sqrt(eps(T)), itmax::Integer = 0) where {T}
    Xm, Bm = _as_matrix(X), _as_matrix(B)
    s = size(Bm, 2)
    size(ws.Bhat, 2) == s || throw(DimensionMismatch("workspace was built for $(size(ws.Bhat, 2)) right-hand sides"))

    # consistent right-hand side Π B and compatibility defect
    _colnorms!(ws.rnorm, Bm, ws)
    nB² = sum(abs2, ws.rnorm)
    copyto!(ws.Bhat, Bm)
    _project!(ws.Bhat, ws)
    _colnorms!(ws.bnorm, ws.Bhat, ws)
    defect = nB² > 0 ? sqrt(max(nB² - sum(abs2, ws.bnorm), zero(T)) / nB²) : zero(T)

    # rank-deficient blocks: Krylov.jl's block MINRES needs a full-rank block, so solve for an
    # orthonormal basis Q of span(Π B) = Q C and recombine X = Y C (rare; allocates)
    C = _rank_compress(ws.Bhat)
    if C !== nothing
        Q, Cm = C
        if size(Q, 2) == 0                              # Π B = 0: the solution is zero
            fill!(Xm, zero(T))
            return BlockMinresStats(true, 0, "zero right-hand side", defect)
        end
        sub = ProjectedMinresWorkspace_from(ws, size(Q, 2))
        Y = similar(Q)
        stats = _pbminres_core!(Y, sub, A, Q; atol, rtol, itmax)
        mul!(Xm, Y, copyto!(similar(Q, size(Cm)...), Cm))
        _ground!(Xm, ws)
        return BlockMinresStats(stats.converged, stats.iterations, stats.status, defect)
    end
    stats = _pbminres_core!(Xm, ws, A, ws.Bhat; atol, rtol, itmax)
    return BlockMinresStats(stats.converged, stats.iterations, stats.status, defect)
end

# Solve A X = Bc for a consistent, full-rank block Bc (may alias ws.Bhat) and ground X.
function _pbminres_core!(Xm, ws::ProjectedMinresWorkspace, A, Bc; atol, rtol, itmax)
    Bhat = ws.Bhat
    Bhat === Bc || copyto!(Bhat, Bc)
    op = A
    if ws.d !== nothing
        Bhat .*= ws.d                                   # D^{-1/2} b
        op = _ScaledOperator{eltype(Bhat), typeof(A), typeof(ws.d), typeof(ws.buf)}(A, ws.d, ws.buf)
    end
    if ws.kw isa Krylov.BlockMinresWorkspace
        Krylov.block_minres!(ws.kw, op, Bhat; atol, rtol, itmax)
        copyto!(Xm, ws.kw.X)
        st = ws.kw.stats
        converged, iterations, status = st.solved, st.niter, st.status
    else                                                # one column at a time
        converged, iterations, status = true, 0, "solution good enough given atol and rtol"
        for j in axes(Bhat, 2)
            Krylov.minres!(ws.kw, op, view(Bhat, :, j); atol, rtol, itmax)
            copyto!(view(Xm, :, j), ws.kw.x)
            st = ws.kw.stats
            converged &= st.solved
            iterations = max(iterations, st.niter)
            st.solved || (status = st.status)
        end
    end
    ws.d === nothing || (Xm .*= ws.d)                   # x = D^{-1/2} y
    _ground!(Xm, ws)
    return (converged = converged, iterations = iterations, status = status)
end

# Orthonormal basis of span(B) if B is (numerically) rank deficient: returns (Q, C) with B ≈ Q C,
# otherwise nothing. Uses the s × s Gram matrix on the host.
function _rank_compress(B::AbstractMatrix{T}; tol = sqrt(eps(T))) where {T}
    s = size(B, 2)
    G = copyto!(Matrix{T}(undef, s, s), B' * B)
    E = eigen(Symmetric(G))
    λmax = maximum(E.values; init = zero(T))
    keep = findall(>(tol * λmax), E.values)
    length(keep) == s && return nothing
    U = E.vectors[:, keep]
    Λ = E.values[keep]
    Q = B * copyto!(similar(B, s, length(keep)), U ./ sqrt.(Λ)')    # orthonormal columns
    C = (sqrt.(Λ) .* U')                                              # r × s, B = Q C
    return (Q, C)
end

# workspace with the same projection data for r right-hand sides
function ProjectedMinresWorkspace_from(ws::ProjectedMinresWorkspace{T}, r::Integer) where {T}
    n = size(ws.V, 1)
    proto = similar(ws.Bhat, n, max(r, 1))
    SV = typeof(similar(proto, n))
    kw = ws.kw isa Krylov.BlockMinresWorkspace ? Krylov.BlockMinresWorkspace(n, n, max(r, 1), SV, typeof(proto)) :
                                                 Krylov.MinresWorkspace(n, n, SV)
    k = size(ws.V, 2)
    return ProjectedMinresWorkspace(kw, similar(proto), similar(proto), ws.d, ws.V, ws.W, ws.F,
                                    fill!(similar(proto, k, max(r, 1)), 0), fill!(similar(proto, 1, max(r, 1)), 0),
                                    zeros(T, r), zeros(T, r), zeros(T, r))
end

"""
    X, stats = pbminres(A, B; nullspace = nothing, grounding = nothing, scaling = :jacobi, kwargs...)

Allocating convenience wrapper around [`ProjectedMinresWorkspace`](@ref) and [`pbminres!`](@ref).
"""
function pbminres(A, B::AbstractVecOrMat; nullspace = nothing, grounding = nothing,
                  scaling::Symbol = :jacobi, diagonal = nothing, columnwise::Bool = false, kwargs...)
    ws = ProjectedMinresWorkspace(A, B; nullspace, grounding, scaling, diagonal, columnwise)
    X = similar(B)
    stats = pbminres!(X, ws, A, B; kwargs...)
    return X, stats
end
