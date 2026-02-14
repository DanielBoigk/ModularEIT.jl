using LinearAlgebra
using SparseArrays
using LinearMaps
using IterativeSolvers

mutable struct GaussNewtonState
    J::AbstractArray
    r::AbstractVector
    δ::AbstractVector
    M::LinearMaps.WrappedMap
end
function GaussNewtonState(n::Int64,k::Int64) # Just some default constructor
    J = zeros(k,n)
    r = zeros(k)
    δ = zeros(n)
    M = LinearMap(spdiagm(ones(n)))
    GaussNewtonState(J,r,δ,M)
end

function gauss_newton_cg!(gns::GaussNewtonState, λ::Float64 = 1e-3, maxiter = 500)
    J = gns.J
    r = gns.r
    M = gns.M
    J_map = LinearMap(J)
    if λ ≠ 0.0
        A_map = J_map' * J_map + λ * M
    else
        A_map = J_map' * J_map
    end
    b = -(J' * r)
    cg!(gns.δ, A_map, b; maxiter = maxiter)
    gns.δ
end
# for reference with svd but only with Levenberg Marquardt
function gauss_newton_svd(J::Matrix{Float64}, r::Vector{Float64}; λ::Float64=1e-3)
    U, Σ, V = svd(J, full=false)
    n = length(Σ)
    Σ_damped = zeros(n)
    for i in 1:n
        Σ_damped[i] = Σ[i] / (Σ[i]^2 + λ) # Levenberg-Marquardt regularization
    end
    -V * (Σ_damped .* (U' * r))
end
function gauss_newton_svd!(gns::GaussNewtonState, λ::Float64=1e-3)
    J = gns.J
    r = gns.r
    U, Σ, V = svd(J, full=false)
    n = length(Σ)
    Σ_damped = zeros(n)
    for i in 1:n
        Σ_damped[i] = Σ[i] / (Σ[i]^2 + λ) # Levenberg-Marquardt regularization
    end
    gns.δ = -V * (Σ_damped .* (U' * r))
end
function update_M!(gns::GaussNewtonState, regularizers::Tuple{Float64, <:AbstractMatrix}...)
    if isempty(regularizers)
        # Handle case with no regularizers, e.g., set M to a zero map
        n = size(gns.J, 2)
        gns.M = LinearMap(zeros(n, n))
        return
    end

    n = size(regularizers[1][2], 1)
    M_sum = zeros(n, n)

    # Calculate the weighted sum: M_sum = Σ (λᵢ * Mᵢ)
    for (λ, M) in regularizers
        @assert size(M) == (n, n) "All M matrices must have the same dimensions."
        M_sum .+= λ .* M
    end
    gns.M = LinearMap(M_sum, issymmetric=true)
end
