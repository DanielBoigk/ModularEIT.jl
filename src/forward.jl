"""
    ForwardProblem(mesh, electrodes)

The EIT forward problem: given a conductivity ``\\sigma`` and injected
currents ``I``, find the electric potential ``u`` with

```math
\\nabla \\cdot (\\sigma \\nabla u) = 0 \\quad \\text{in } \\Omega .
```

See also [`solve_forward`](@ref), [`jacobian`](@ref).
"""
struct ForwardProblem{M<:EITMesh,E<:Electrode}
    mesh::M
    electrodes::Vector{E}
end

"""
    solve_forward(problem::ForwardProblem, σ::AbstractVector, I::AbstractVector) -> Vector

Return the electrode voltages for the element-wise conductivity `σ` and the
current pattern `I` (one entry per electrode, summing to zero).

This mock uses a diagonal "resistor" model instead of finite elements.
"""
function solve_forward(problem::ForwardProblem, σ::AbstractVector, I::AbstractVector)
    L = length(problem.electrodes)
    length(I) == L || throw(DimensionMismatch("expected $L currents, got $(length(I))"))
    length(σ) == nelements(problem.mesh) ||
        throw(DimensionMismatch("expected one conductivity value per element"))
    z = [e.impedance for e in problem.electrodes]
    U = I ./ mean_conductivity(σ) .+ z .* I
    return U .- sum(U) / L
end

"""
    jacobian(problem::ForwardProblem, σ::AbstractVector, I::AbstractVector) -> Matrix

Finite-difference sensitivity matrix ``J_{\\ell k} = \\partial U_\\ell / \\partial \\sigma_k``.
"""
function jacobian(problem::ForwardProblem, σ::AbstractVector, I::AbstractVector; h=1e-6)
    U0 = solve_forward(problem, σ, I)
    J = zeros(length(U0), length(σ))
    for k in eachindex(σ)
        σk = copy(σ)
        σk[k] += h
        J[:, k] = (solve_forward(problem, σk, I) - U0) / h
    end
    return J
end

mean_conductivity(σ) = sum(σ) / length(σ)
