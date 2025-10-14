# https://www.scirp.org/pdf/eng_2022032413543094.pdf

```julia
using LinearAlgebra
using SparseArrays

"""
    anisotropic_tv_admm(S, delta, lambda, alpha; n, maxiter=100, tol=1e-4)

Solves the anisotropic Total Variation (TV) regularized inverse problem using
the Alternating Direction Method of Multipliers (ADMM):

```
min_σ 1/2 ‖S σ - δ‖₂² + λ (‖Dₓ σ‖₁ + ‖Dᵧ σ‖₁)
```

where σ ∈ ℝᴺ is the vectorized image (N = n²), S is the forward operator (m × N),
δ is the observed data (m-vector), λ > 0 is the regularization parameter,
Dₓ and Dᵧ are the N × N first-order finite difference matrices in the horizontal
and vertical directions (with zero boundary conditions), and α > 0 is the ADMM
penalty parameter.

The problem is reformulated as a constrained minimization:

```
min_{σ, u} 1/2 ‖S σ - δ‖₂² + λ ‖u‖₁  s.t.  u = D σ
```

with D = [Dₓ; Dᵧ] ∈ ℝ^{2N × N}, solved via ADMM iterations:

- σ^{k+1} = arg min_σ [1/2 ‖S σ - δ‖₂² + (α/2) ‖D σ - uᵏ + vᵏ/α‖₂²]
  (solved via linear system (SᵀS + α DᵀD) σ = Sᵀδ + α Dᵀ(uᵏ - vᵏ/α))

- u^{k+1} = soft_threshold(D σ^{k+1} - vᵏ/α, λ/α)
  (component-wise soft-thresholding: sign(z) ⊙ max(|z| - τ, 0) with τ = λ/α)

- v^{k+1} = vᵏ + α (D σ^{k+1} - u^{k+1})

Initialization: σ⁰ = u⁰ = v⁰ = 0.

Convergence is checked via relative change in σ: ‖σ^{k+1} - σᵏ‖₂ / ‖σᵏ‖₂ < tol.

# Arguments
- `S::AbstractMatrix`: Forward operator (m × N).
- `delta::AbstractVector`: Observed data (length m).
- `lambda::Real`: TV regularization parameter (> 0).
- `alpha::Real`: ADMM penalty parameter (> 0).
- `n::Int`: Image dimension (√N = n).
- `maxiter::Int`: Maximum number of ADMM iterations (default: 100).
- `tol::Real`: Relative convergence tolerance (default: 1e-4).

# Returns
- `sigma::Vector{Float64}`: Reconstructed image (length N).

# Notes
- Dₓ and Dᵧ use forward finite differences with zero boundaries (last row/column differences are zero).
- For large N, preconditioning or iterative solvers (e.g., CG) may be needed for the σ update.
- Reference: Adapted from anisotropic TV-ADMM scheme in [Wang, Engineering, 2022].
"""
function anisotropic_tv_admm(S::AbstractMatrix, delta::AbstractVector, lambda::Real, alpha::Real;
                             n::Int, maxiter::Int=100, tol::Real=1e-4)
    N = size(S, 2)
    @assert N == n^2 "Image dimension mismatch: size(S, 2) must equal n²"
    m = size(S, 1)
    @assert length(delta) == m "Data dimension mismatch: length(delta) must equal size(S, 1)"

    D = build_derivative_operators(n)
    sigma = zeros(N)
    u = zeros(2 * N)
    v = zeros(2 * N)

    for iter in 1:maxiter
        # σ update: solve (SᵀS + α DᵀD) σ = Sᵀδ + α Dᵀ(u - v/α)
        STS = S' * S
        DTD = D' * D
        A = STS + alpha * DTD
        b = S' * delta + alpha * (D' * (u - v / alpha))
        sigma_new = A \ b

        # u update: soft-thresholding
        z = D * sigma_new - v / alpha
        tau = lambda / alpha
        u_new = sign.(z) .* max.(abs.(z) .- tau, 0.)

        # v update
        v .+= alpha * (D * sigma_new - u_new)

        # Convergence check
        rel_change = norm(sigma_new - sigma) / (norm(sigma) + 1e-10)
        sigma = sigma_new
        u = u_new
        if rel_change < tol
            @info "Converged after $iter iterations (rel_change = $rel_change)"
            break
        end
    end

    return sigma
end

function build_derivative_operators(n::Int)
    N = n * n

    # Horizontal differences Dₓ: forward diff, zero on last column
    I_x = Int[]
    J_x = Int[]
    V_x = Float64[]
    for i in 1:n
        for j in 1:(n-1)
            row = (i - 1) * n + j
            append!(I_x, [row, row])
            append!(J_x, [row, row + 1])
            append!(V_x, [-1.0, 1.0])
        end
    end
    Dx = sparse(I_x, J_x, V_x, N, N)

    # Vertical differences Dᵧ: forward diff, zero on last row
    I_y = Int[]
    J_y = Int[]
    V_y = Float64[]
    for i in 1:(n-1)
        for j in 1:n
            row = i * n + j
            append!(I_y, [row, row])
            append!(J_y, [row - n, row])
            append!(V_y, [1.0, -1.0])  # Note: sign flipped for consistency (∂/∂y = σ_{i+1,j} - σ_{i,j})
        end
    end
    Dy = sparse(I_y, J_y, V_y, N, N)

    # Stack: D = [Dx; Dy]
    D = vcat(Dx, Dy)
    return D
end
