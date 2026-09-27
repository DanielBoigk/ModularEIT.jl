"""
    ReconstructionResult

Output of [`reconstruct`](@ref).

# Fields
- `σ::Vector{Float64}`: reconstructed conductivity.
- `residuals::Vector{Float64}`: objective value after each iteration.
- `converged::Bool`: whether the gradient tolerance was reached.
"""
struct ReconstructionResult
    σ::Vector{Float64}
    residuals::Vector{Float64}
    converged::Bool
end

"""
    reconstruct(problem, U, I, R::Regularizer; σ_init, iterations=100, step=1e-2, tol=1e-8)

Recover the conductivity from measured voltages `U` for the current pattern `I`
by gradient descent on

```math
\\min_\\sigma \\; \\tfrac12 \\lVert F(\\sigma) - U \\rVert_2^2 + R(\\sigma).
```

# Examples
```julia
mesh  = circle_mesh(32)
prob  = ForwardProblem(mesh, ring_electrodes(mesh, 8))
I     = [1.0, -1.0, 0, 0, 0, 0, 0, 0]
U     = solve_forward(prob, fill(2.0, nelements(mesh)), I)
res   = reconstruct(prob, U, I, Tikhonov(1e-3); σ_init=ones(nelements(mesh)))
```
"""
function reconstruct(problem::ForwardProblem, U::AbstractVector, I::AbstractVector,
                     R::Regularizer; σ_init::AbstractVector,
                     iterations::Integer=100, step::Real=1e-2, tol::Real=1e-8)
    σ = float.(collect(σ_init))
    residuals = Float64[]
    for _ in 1:iterations
        r = solve_forward(problem, σ, I) - U
        push!(residuals, sum(abs2, r) / 2 + penalty(R, σ))
        g = jacobian(problem, σ, I)' * r + gradient(R, σ)
        norm(g) < tol && return ReconstructionResult(σ, residuals, true)
        σ .-= step .* g
    end
    return ReconstructionResult(σ, residuals, false)
end
