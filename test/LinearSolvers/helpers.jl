using LinearAlgebra
using SparseArrays
using Random

# Weighted 5-point graph Laplacian on an m×m grid: symmetric positive semidefinite,
# null space = constants. Same structure as ∫ σ ∇φᵢ⋅∇φⱼ dΩ with a pure Neumann boundary.
function neumann_laplacian(m; rng = Random.default_rng(), contrast = 10.0)
    idx(i, j) = (j - 1) * m + i
    I, J, V = Int[], Int[], Float64[]
    for j in 1:m, i in 1:m, (di, dj) in ((1, 0), (0, 1))
        (i + di > m || j + dj > m) && continue
        a, b = idx(i, j), idx(i + di, j + dj)
        w = 1 + (contrast - 1) * rand(rng)
        append!(I, (a, b, a, b)); append!(J, (a, b, b, a)); append!(V, (w, w, -w, -w))
    end
    return sparse(I, J, V, m^2, m^2)
end

boundary_dofs(m) = [(j - 1) * m + i for j in 1:m, i in 1:m if i in (1, m) || j in (1, m)]

# Right-hand sides compatible with the Neumann problem (current only on the boundary, mean zero).
function boundary_currents(m, s)
    bd = boundary_dofs(m)
    B = zeros(m^2, s)
    for k in 1:s
        θ = range(0, 2π; length = length(bd) + 1)[1:end-1]
        B[bd, k] .= isodd(k) ? cos.(((k + 1) ÷ 2) .* θ) : sin.((k ÷ 2) .* θ)
        B[:, k] .-= sum(B[:, k]) / m^2
    end
    return B
end
