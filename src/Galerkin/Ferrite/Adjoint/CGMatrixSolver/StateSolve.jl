function conjugate_gradient_reference(
    K, b::AbstractVector, x0::AbstractVector=zero(b); atol=length(b) * eps(norm(b))
)
    L = state.L
    x = copy(x0)                        # initialize the solution
    r = b - A * x0                      # initial residual
    p = copy(r)                         # initial search direction
    r²old = r' * r                      # squared norm of residual

    k = 0
    while r²old > atol^2                # iterate until convergence
        Ap = A * p                      # search direction
        α = r²old / (p' * Ap)           # step size
        @. x += α * p                   # update solution
        # Update residual:
        if (k + 1) % 16 == 0            # every 16 iterations, recompute residual from scratch
            r .= b .- A * x             # to avoid accumulation of numerical errors
        else
            @. r -= α * Ap              # use the updating formula that saves one matrix-vector product
        end
        r²new = r' * r
        @. p = r + (r²new / r²old) * p  # update search direction
        r²old = r²new                   # update squared residual norm
        k += 1
    end

    return x
end

function state_BlockCG_step(cgnm::CGNeumannMAtrix, atol::Number, max_iter::Number=1000)

    lamd = cgnm.L₀
    A = lamd.L

    b = cgnm.u_rhs::AbstractArray # preconditioned righhandside state equation
    x = cgnm.u #AbstractArray
    r = cgnm.r #AbstractVector # the residual
    error = cgnm.error #Number # sum of residuals

    # For CG solve
    cgnm.u_r #Union{AbstractArray,Nothing}
    p = cgnm.u_p #Union{AbstractArray,Nothing}
    r²old = cgnm.u_r²old #AbstractVector
    r²new = cgnm.u_r²new #AbstractVector
    Ap = cgnm.u_Ap #Union{AbstractArray,Nothing}
    α = cgnm.u_α #AbstractVector

    for i in 1:max_iter
        if r²old > atol^2 # This needs to be converted still to matrix version.
            error = sum(r²old)
            return error
        end
        mul!(Ap, A, p) #Ap = A * p # search direction
        α = r²old / (p' * Ap)           # step size
        @. x += α * p                   # update solution
        # Update residual:
        if (k + 1) % 16 == 0            # every 16 iterations, recompute residual from scratch
            r .= b
            mul!(r, A, x, -1.0, 1.0)
            #r .= b .- A * u             # to avoid accumulation of numerical errors
        else
            @. r -= α * Ap              # use the updating formula that saves one matrix-vector product
        end
        r²new = r' * r#r²new = r' * r
        @. p = r + (r²new / r²old) * p  # update search direction
        r²old = r²new                   # update squared residual norm
        k += 1

    end
    error = sum(r²old)
    return error
end

# Non allocating Block CG version specifically for adjoint solve.
function state_BlockCG_step!(cgnm::CGNeumannMatrix, atol::Number, max_iter::Number=1000)
    A = cgnm.L₀.L
    B = cgnm.u_rhs   # n×m right-hand side matrix
    X = cgnm.u       # n×m solution matrix
    R = cgnm.r       # n×m residual matrix
    P = cgnm.u_p     # n×m search direction matrix
    Ap = cgnm.u_Ap    # n×m matrix: A*P

    # m×m matrices — must be pre-allocated in CGNeumannMatrix
    RtR_old = cgnm.u_r²old   # R'R
    RtR_new = cgnm.u_r²new   # updated R'R
    PtAp = cgnm.u_PtAp    # P'AP  (m×m, pre-allocate in struct)
    α = cgnm.u_α       # m×m step size matrix
    β = cgnm.u_β       # m×m direction update matrix (pre-allocate in struct)

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
