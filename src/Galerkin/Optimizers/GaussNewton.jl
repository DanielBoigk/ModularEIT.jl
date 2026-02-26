using LinearAlgebra
using SparseArrays
using LinearMaps
using IterativeSolvers
using Base: *

export gauss_newton_lm_cg!, gauss_newton_svd!, gauss_newton_lm_lsqr!, proximal_gauss_newton_step!
"""
    gauss_newton_cg!(opt::GalerkinOptState; maxiter=500)

Compute one Gauss–Newton or Levenberg–Marquardt update using the **Conjugate Gradient (CG)** method.

Forms the (possibly regularized) normal equations
```math
(J^T J + λ L)\\, δ = -J^T r,
```
and solves them approximately by CG.

# Arguments
- `opt` — Optimization state containing `J`, `r`, `L`, `λ`, and current `δ`.
- `maxiter` — Maximum number of CG iterations (default: 500).

# Returns
The updated parameter increment `δ`.

# Notes
- If `λ ≠ 0.0`, Levenberg–Marquardt damping is applied via `A = J'J + λL`.
- Supports matrix-free operation when `J` or `L` are given as `LinearMap`s.
- The solution is written back into `opt.δ` in place.
"""
function gauss_newton_lm_cg!(opt::GalerkinOptState, maxiter=500)
    J = opt.J
    r = opt.r
    L = opt.L
    λ = opt.λ
    δ = opt.δ
    J_map = LinearMap(J)
    if λ ≠ 0.0
        A_map = J_map' * J_map + λ * L
    else
        A_map = J_map' * J_map
    end
    b = -(J' * r)
    cg!(δ, A_map, b; maxiter=maxiter)
    δ .*= opt.τ
end


"""
    gauss_newton_svd(opt::GalerkinOptState)

Compute a **Levenberg–Marquardt–regularized Gauss–Newton step** using
the singular value decomposition (SVD) of `J`.

Performs the update
```math
δ = -V \\, \\mathrm{diag}\\!\\left(\\frac{Σ_i}{Σ_i^2 + λ}\\right) U^T r,
```
which corresponds to **LM damping with `L = I`**.

# Warning
This method assumes that the regularization operator `L` is the identity.
If a different matrix `L` is used (e.g. for smoothness or curvature
regularization), the SVD-based formula is no longer valid.
Use [`gauss_newton_cg!`] instead, which can handle arbitrary `L`.

# Arguments
- `opt` — Optimization state containing `J`, `r`, and `λ`.

# Returns
Updates `opt.δ` in place with the computed step.

# Notes
- `λ → 0` recovers standard Gauss–Newton.
- Large `λ` approaches gradient descent.
- Suitable only for small or dense `J`.
"""
function gauss_newton_svd!(opt::FerriteOptState)
    J = opt.J
    r = opt.r
    λ = opt.λ
    U, Σ, V = LinearAlgebra.svd(J)
    n = length(Σ)
    Σ_damped = zeros(n)
    for i in 1:n
        Σ_damped[i] = Σ[i] / (Σ[i]^2 + λ) # Levenberg-Marquardt regularization
    end
    opt.δ = -V * (Σ_damped .* (U' * r))
end

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
    lsqr!(δ, A_map, b; maxiter=maxiter, atol=tol, btol=tol)
    return δ
end

# Fix still:
function proximal_gauss_newton_step!(opt::FerriteOptState, σ::AbstractVector; maxiter=500)
    J = LinearMap(opt.J)
    λ = opt.λ
    β_d = opt.β_diff
    β_nd = opt.β_ndiff
    τ = opt.τ # Step size / learning rate

    # Differentiable regularizer gradient
    grad_smooth = opt.J' * opt.r + β_d * opt.∇R(σ)

    # Solve Gauss Newton
    A_map = J' * J + λ * I
    cg!(opt.δ, A_map, -grad_smooth; maxiter=maxiter)

    # update standard step
    σ_trial = σ + τ * opt.δ

    # proximal step
    threshold = τ * β_nd

    # and then clamping for your 1e-6 physical constraint
    for i in eachindex(σ)
        # Apply L1 proximal operator
        val = σ_trial[i]
        # Find out what the exact proximal operator is:
        σ_new = sign(val) * max(abs(val) - threshold, 0.0)

        # Apply Box Constraint (Non-negativity)
        opt.δ[i] = max(σ_new, 1e-6) - σ[i]
    end
end
