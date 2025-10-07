using LinearAlgebra
using IterativeSolvers
using Statistics

export EITModeN
export state_adjoint_step_neumann_cg!
export objective!
export gradient!

mutable struct EITModeN
    u::AbstractVector
    b::AbstractVector
    λ::AbstractVector
    δσ::AbstractVector
    f::AbstractVector
    g::AbstractVector
    rhs::AbstractVector
    error::Float64
    length::Int64
    m::Int64
end
function EITModeN(g::AbstractVector, f::AbstractVector)
    L = length(g)
    M = length(f)
    return EITModeN(zeros(L), zeros(M), zeros(L), zeros(L), f, g, zeros(L), 0.0, L, M)
end



function state_adjoint_step_neumann_cg!(mode::EITModeN, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    d = sol.d
    ∂d = sol.∂d
    L = sol.L
    down = fe.down
    up = fe.up
    # We solve the state equation ∇⋅(σ∇uᵢ) = 0 : σ∂u/∂𝐧 = g
    cg!(mode.u, L, mode.g; maxiter=maxiter)
    mode.b = down(mode.u)
    mean = Statistics.mean(mode.b)
    mode.b .-= mean
    mode.u .-= mean
    # We solve the adjoint equation ∇⋅(σ∇λᵢ) = 0 : σ∂λ/∂𝐧 = ∂ₓd(u,f)
    cg!(mode.λ, L, up(∂d(mode.b, mode.f)); maxiter=maxiter)
    # Calculate J(σ,f,g)
    mode.error = d(mode.b, mode.f)
    # Calculate ∂J(σ,f,g)/∂σ = ∇(uᵢ)⋅∇(λᵢ) here:
    mode.δσ = calculate_bilinear_map!(fe,mode.rhs, mode.λ, mode.u)
    return mode.δσ, mode.error
end

function objective!(mode::EITModeN, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    d = sol.d
    ∂d = sol.∂d
    L = sol.L
    down = fe.down
    up = fe.up
    # We solve the state equation ∇⋅(σ∇uᵢ) = 0 : σ∂u/∂𝐧 = g
    cg!(mode.u, L, mode.g; maxiter=maxiter)
    mode.b = down(mode.u)
    mean = Statistics.mean(mode.b)
    mode.b .-= mean
    mode.u .-= mean
    mode.error = d(mode.b, mode.f)
    return mode.error
end

function gradient!(mode::EITModeN, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    d = sol.d
    ∂d = sol.∂d
    L = sol.L
    down = fe.down
    up = fe.up
    # We solve the adjoint equation ∇⋅(σ∇λᵢ) = 0 : σ∂λ/∂𝐧 = ∂ₓd(u,f)
    cg!(mode.λ, L, up(∂d(mode.b, mode.f)); maxiter=maxiter)
    # Calculate ∂J(σ,f,g)/∂σ = ∇(uᵢ)⋅∇(λᵢ) here:
    mode.δσ = calculate_bilinear_map!(fe,mode.rhs, mode.λ, mode.u)
    return mode.δσ
end
