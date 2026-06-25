using LinearAlgebra, LinearMaps, Base
# Write CG as a block veriosn from ground up.


"""
Holds all preallocated arrays for a single Block CG solve of AX = B.
n = rows of A, m = number of RHS columns.
"""
struct BlockCGIterator{T,M<:AbstractMatrix{T},P}
    # Problem
    A::M          # n×n system matrix (reference, not owned)
    B::M          # n×m right-hand side (reference, not owned)
    X::M          # n×m solution

    # n×m working arrays
    R::M          # residual
    P::M          # search direction
    Ap::M          # A*P scratch

    # m×m working arrays
    RtR_old::M     # R'R from previous iteration
    RtR_new::M     # R'R current iteration
    PtAp::M     # P'AP
    α::M     # step size
    β::M     # direction update
    scratch::M     # spare m×m to avoid allocs in ldiv!

    # Scalar state
    k::Base.RefValue{Int}     # iteration counter
    converged::Base.RefValue{Bool}


    Z::M    # n×m preconditioned residual: Z = M⁻¹R
    preconditioner::P   # any type implementing ldiv!(Z, M, R)
end

function BlockCGIterator(A::M, B::M, X::M) where {T,M<:AbstractMatrix{T}}
    n, m = size(B)
    @assert size(A) == (n, n)
    @assert size(X) == (n, m)

    return BlockCGIterator(
        A, B, X,
        similar(B),         # R
        similar(B),         # P
        similar(B),         # Ap
        similar(B, m, m),   # RtR_old
        similar(B, m, m),   # RtR_new
        similar(B, m, m),   # PtAp
        similar(B, m, m),   # α
        similar(B, m, m),   # β
        similar(B, m, m),   # scratch
        Ref(0),
        Ref(false),
    )
end

function initialize!(bcg::BlockCGIterator)
    # R = B - A*X
    copy!(bcg.R, bcg.B)
    mul!(bcg.R, bcg.A, bcg.X, -1.0, 1.0)
    # P = R
    copy!(bcg.P, bcg.R)
    # RtR_old = R'R
    mul!(bcg.RtR_old, bcg.R', bcg.R)
    bcg.k[] = 0
    bcg.converged[] = false
end

function state_BlockCG_step!(bcg::BlockCGIterator, atol::Number, max_iter::Number=1000)
    A = bcg.L₀.L
    B = bcg.u_rhs   # n×m right-hand side matrix
    X = bcg.u       # n×m solution matrix
    R = bcg.r       # n×m residual matrix
    P = bcg.u_p     # n×m search direction matrix
    Ap = bcg.u_Ap    # n×m matrix: A*P

    # m×m matrices — must be pre-allocated in CGNeumannMatrix
    RtR_old = bcg.u_r²old   # R'R
    RtR_new = bcg.u_r²new   # updated R'R
    PtAp = bcg.u_PtAp    # P'AP  (m×m, pre-allocate in struct)
    α = bcg.u_α       # m×m step size matrix
    β = bcg.u_β       # m×m direction update matrix (pre-allocate in struct)

    atol2 = atol^2

    for k in 1:max_iter
        # Check convergence: norm of R'R as proxy for ‖R‖²_F
        if norm(RtR_old) < atol2
            return sum(diag(RtR_old))  # return sum of squared residual norms
        end

        # Ap = A * P
        mul!(Ap, A, P)

        # PtAp = P' * Ap  (m×m, no alloc: Adjoint is lazy)
        mul!(PtAp, P', Ap)

        # α = RtR_old / PtAp  →  solve PtAp' * α' = RtR_old'
        # i.e. α = RtR_old * inv(PtAp), done via in-place ldiv after factorisation
        copy!(α, RtR_old)
        ldiv!(lu!(PtAp), α)    # α ← PtAp \ RtR_old  (modifies PtAp — refactor if needed)

        # X += P * α
        mul!(X, P, α, 1.0, 1.0)

        # Residual update
        if k % 16 == 0
            # Recompute from scratch to avoid floating point drift
            copy!(R, B)
            mul!(R, A, X, -1.0, 1.0)   # R = B - A*X
        else
            # R -= Ap * α
            mul!(R, Ap, α, -1.0, 1.0)
        end

        # RtR_new = R' * R  (m×m)
        mul!(RtR_new, R', R)

        # β = inv(RtR_old) * RtR_new  →  solve RtR_old * β = RtR_new
        copy!(β, RtR_new)
        ldiv!(lu!(copy(RtR_old)), β)   # β ← RtR_old \ RtR_new

        # P = R + P * β
        mul!(P, P, β, 1.0, 0.0)        # P ← P * β
        axpy!(1.0, R, P)               # P ← R + P

        # Swap old/new (just copy — both are pre-allocated)
        copy!(RtR_old, RtR_new)
    end

    return sum(diag(RtR_old))
end
