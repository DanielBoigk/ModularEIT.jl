using Statistics
using IterativeSolvers
using LinearAlgebra
using Ferrite

mutable struct EITModeM
    u_g::AbstractVector
    u_g::AbstractVector
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
function EITModeM(g::AbstractVector, f::AbstractVector,m::Int64)
    length = length(g)
    return EITModeM(zeros(length), zeros(length), zeros(length), zeros(length), zeros(length), f, g, zeros(length), 0.0, length, m)
end


function state_adjoint_step_mixed_cg!(mode::EITModeM, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    Ln = sol.L
    Ld = sol.LD
    n = sol.n
    ∂n = sol.∂n
    down = sol.down
    up = sol.up
    # We solve the state equation ∇⋅(σ∇uᵢ) = 0 : σ∂u/∂𝐧 = g
    cg!(mode.u_g, Ln, mode.g; maxiter=maxiter)
    # We solve the state equation ∇⋅(σ∇uᵢ) = 0 : u = f
    cg!(mode.u_f, Ld, mode.f; maxiter=maxiter)

    # Normalize over the boundary:
    b = down(mode.u_g)
    mean = Statistics.mean(b)
    mode.u_g .-= mean
    b = down(mode.u_f)
    mean = Statistics.mean(b)
    mode.u_f .-= mean


    mode.w = mode.u_f - mode.u_g


    # We solve the adjoint equation ∇⋅(σ∇λᵢ) = ∂ₓd(w,0)
    cg!(mode.λ, Ln, ∂n(mode.w); maxiter=maxiter)
    mode.error = n(mode.w)
    # Calculate ∇(uᵢ)⋅∇(λᵢ) here:
    mode.δσ = calculate_bilinear_map!(fe,mode.rhs, mode.λ, mode.w)
    return mode.δσ, mode.error
end

function objective!(mode::EITModeM, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    Ln = sol.L
    Ld = sol.LD
    n = sol.n
    ∂n = sol.∂n
    down = sol.down
    up = sol.up
    # We solve the state equation ∇⋅(σ∇uᵢ) = 0 : σ∂u/∂𝐧 = g
    cg!(mode.u_g, Ln, mode.g; maxiter=maxiter)
    # We solve the state equation ∇⋅(σ∇uᵢ) = 0 : u = f
    cg!(mode.u_f, Ld, mode.f; maxiter=maxiter)

    # Normalize over the boundary:
    b = down(mode.u_g)
    mean = Statistics.mean(b)
    mode.u_g .-= mean
    b = down(mode.u_f)
    mean = Statistics.mean(b)
    mode.u_f .-= mean


    mode.w = mode.u_f - mode.u_g

    mode.error = n(mode.w)
    return mode.error
end

function gradient!(mode::EITModeM, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    Ln = sol.L
    Ld = sol.LD
    n = sol.n
    ∂n = sol.∂n
    down = sol.down
    up = sol.up
    # We solve the adjoint equation ∇⋅(σ∇λᵢ) = ∂n(w)
    cg!(mode.λ, Ln, ∂n(mode.w); maxiter=maxiter)
    # Calculate ∇(uᵢ)⋅∇(λᵢ) here:
    mode.δσ = calculate_bilinear_map!(fe,mode.rhs, mode.λ, mode.w)
    return mode.δσ
end
