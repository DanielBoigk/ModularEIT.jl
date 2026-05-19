# objective:
# prox(y) = argmin_z λTK(z) + ρ/2 * ||z-y||^2
# prox(y) = argmin_z λz^T Mz + ρ/2 * ||z-y||^2
# ∇_z = λMz + ρ (z-y)
# λMz = ρ(y-z)   
# (λM + ρI)z = ρy

using LinearMaps, IterativeSolvers, LinearAlgebra

export get_prox_Tikhonov

function get_prox_Tikhonov(K::AbstractMatrix, ρ::Number, λ::Number)
    N = size(K, 1)
    # Build (λK + ρI) as a lazy LinearMap — no allocation
    A = LinearMap(z -> λ * (K * z) + ρ * z, N; issymmetric=true, isposdef=true)
    prox = function(y::AbstractVector)
        b = ρ * y
        return cg(A, b; x0=b)   # cg(A, b), not cg(y, A, b)
    end
    return prox
end