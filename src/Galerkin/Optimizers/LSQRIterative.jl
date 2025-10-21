using LinearMaps, IterativeSolvers


# Run this for now. Debug optimized version later
function gauss_newton_lm_lsqr!(opt::GalerkinOptState, maxiter=200; tol=1e-6)
    J = opt.J
    r = opt.r
    L = opt.L
    λ = opt.λ
    δ = opt.δ
    _, nσ = size(J)
    A = [J; sqrt(λ) * L]
    b = vcat(-r, zeros(nσ))
    A_map = LinearMap(A)
    lsqr!(δ,A_map, b; maxiter=maxiter, atol=tol, btol=tol)
    return δ
end

#=
"""
In-place LSQR Gauss–Newton/LM step.

Solves  (JᵗJ + λ Llm) δσ = -Jᵗ r
without assembling J.

Arguments:
  δσ        -- preallocated parameter update (will contain result)
  gradients -- vector of parameter gradients δσₖ (each same length as δσ)
  r         -- vector of residuals rₖ (same length as gradients)
  λ         -- LM damping factor
  Llm_mul!  -- in-place y = Llm * x
  tmpσ, tmpd -- preallocated work buffers (same lengths as δσ and r)
"""
function lsqr_gn_step!(δσ, gradients, r, λ, Llm_mul!, tmpσ, tmpd; maxiter=200, tol=1e-6)
    nσ = length(δσ)
    nd = length(r)
    # --- Define J * x  (data-space) ---
    function J_mul!(y, x)
        @inbounds for k in 1:nd
            y[k] = dot(gradients[k], x)
        end
        return y
    end
    # --- Define Jᵗ * y  (parameter-space) ---
    function Jt_mul!(x, y)
        fill!(x, 0.0)
        @inbounds for k in 1:nd
            @. x += y[k] * gradients[k]
        end
        return x
    end
    # --- Build LinearMap for augmented least squares ---
    function A_mul!(out, x)
        # out = [J; sqrt(λ) R] * x   (conceptual)
        J_mul!(view(out, 1:nd), x)
        Llm_mul!(view(out, nd+1:nσ+nd), x)
        @. out[nd+1:end] *= sqrt(λ)
        return out
    end
function At_mul!(x, y)
    fill!(x, 0.0)
    Jt_mul!(x, view(y, 1:nd))
    @views tmpσ .= y[nd+1:nσ+nd]
    @. tmpσ *= sqrt(λ)
    Llm_mul!(tmpd, tmpσ)  # Now tmpd should be length nσ
    @. x += tmpd
    return x
end
    Amap = LinearMap(A_mul!, At_mul!, nσ + nd, nσ; issymmetric=false)
    # Right-hand side [ -r ; 0 ]
    rhs = similar(tmpσ, nσ + nd)
    @views rhs[1:nd] .= -r
    fill!(view(rhs, nd+1:end), 0.0)
    # LSQR solve in-place
    fill!(δσ, 0.0)
    lsqr!(δσ, Amap, rhs; maxiter=maxiter, atol=tol, btol=tol)
    return δσ
end

function Llm_chol_mul!(out, v, F, tmpL)  # F is lower triangular Cholesky factor
    # out = F' * (F * v)
    ldiv!(F, v, tmpL)      # tmpL = F \ v  (forward solve, in-place)
    mul!(out, adjoint(F), tmpL)  # out = F' * tmpL  (in-place, uses BLAS)
    return out
end
=#
