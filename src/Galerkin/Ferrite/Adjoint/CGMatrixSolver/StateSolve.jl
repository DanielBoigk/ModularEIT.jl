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
