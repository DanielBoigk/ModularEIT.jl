using LinearAlgebra
using SparseArrays
using LinearMaps
using IterativeSolvers
using Base: *

export gauss_newton_lm_cg!, gauss_newton_svd!
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
    opt.δ = V * (Σ_damped .* (U' * r))
end
