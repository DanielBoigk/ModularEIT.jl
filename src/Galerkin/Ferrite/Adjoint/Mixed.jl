using IterativeSolvers
using LinearAlgebra
using Ferrite

export FerriteEITModeM
export state_adjoint_step_mixed_cg!
export objective_mixed_cg!
export gradient_mixed_cg!

function FerriteEITModeM(g::AbstractVector, f::AbstractVector, m::Int64)
    n = fe.n
    m = fe.m
    u_g = zeros(n)
    u_f = zeros(n)
    w = zeros(n)
    b = zeros(m)
    F = fe.up(f)
    G = fe.up(g)
    λ = zeros(n)
    δσ = zeros(n)
    rhs = zeros(n)
    return FerriteEITMode(u_f, u_g, w, b, λ, δσ, F, f, G, g, rhs, 0.0, 0.0, 0.0)
end


function state_adjoint_step_mixed_cg!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    objective_mixed_cg!(mode, sol, fe, maxiter)
    gradient_mixed_cg!(mode, sol, fe, maxiter)
    return mode.δσ, mode.error_n
end

function objective_mixed_cg!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    n = sol.n
    # We solve the state equation ∇⋅(σ∇uᵢ) = 0 : u = f
    objective_dirichlet_cg!(mode, sol, fe, maxiter)
    # We solve the state equation ∇⋅(σ∇uᵢ) = 0 : σ∂u/∂𝐧 = g
    objective_neumann_cg!(mode, sol, fe, maxiter)

    mode.w = mode.u_f - mode.u_g

    mode.error_m = n(mode.w)
    return mode.error_m
end

function gradient_mixed_cg!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    Ln = sol.L
    ∂n = sol.∂n
    # We solve the adjoint equation ∇⋅(σ∇λᵢ) = ∂n(w)
    cg!(mode.λ, Ln, ∂n(mode.w); maxiter=maxiter)
    # Calculate ∇(uᵢ)⋅∇(λᵢ) here:
    mode.δσ = calculate_bilinear_map!(fe, mode.rhs, mode.λ, mode.w)
    return mode.δσ
end
