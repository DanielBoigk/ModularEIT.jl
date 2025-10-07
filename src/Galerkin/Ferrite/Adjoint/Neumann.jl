using LinearAlgebra
using IterativeSolvers

export FerriteEITModeN
export state_adjoint_step_neumann_cg!
export objective_neumann_cg!
export gradient_neumann_cg!


function FerriteEITModeN(g::AbstractVector, f::AbstractVector, fe::FerriteFESpace)
    n = fe.n
    m = fe.m
    u = zeros(n)
    b = zeros(m)
    F = fe.up(f)
    G = fe.up(g)
    λ = zeros(n)
    δσ = zeros(n)
    rhs = zeros(n)
    return FerriteEITMode(nothing, u, nothing, b, λ, δσ, F, f, G, g, rhs, 0.0, 0.0, 0.0)
end

function state_adjoint_step_neumann_cg!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    objective_neumann_cg!(mode, sol, fe, maxiter)
    gradient_neumann_cg!(mode, sol, fe, maxiter)
    return mode.δσ, mode.error_n
end

function objective_neumann_cg!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    d = sol.d
    ∂d = sol.∂d
    L = sol.L
    down = fe.down
    up = fe.up
    # We solve the state equation ∇⋅(σ∇uᵢ) = 0 : σ∂u/∂𝐧 = g
    cg!(mode.u_g, L, mode.G; maxiter=maxiter)
    # Normalize
    mean_boundary!(mode.u_g, mode, down)
    mode.error_n = d(mode.b, mode.f)
    return mode.error_n
end

function gradient_neumann_cg!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    d = sol.d
    ∂d = sol.∂d
    L = sol.L
    down = fe.down
    up = fe.up
    # We solve the adjoint equation ∇⋅(σ∇λᵢ) = 0 : σ∂λ/∂𝐧 = ∂ₓd(u,f)
    cg!(mode.λ, L, up(∂d(mode.b, mode.f)); maxiter=maxiter)
    # Calculate ∂J(σ,f,g)/∂σ = ∇(uᵢ)⋅∇(λᵢ) here:
    mode.δσ = calculate_bilinear_map!(fe, mode.rhs, mode.λ, mode.u_g)
    return mode.δσ
end
