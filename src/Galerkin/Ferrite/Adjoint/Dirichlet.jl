using LinearAlgebra
using IterativeSolvers

export FerriteEITModeN
export state_adjoint_step_dirichlet_cg!
export objective_dirichlet_cg!
export gradient_dirichlet_cg!


function FerriteEITModeD(g::AbstractVector, f::AbstractVector, fe::FerriteFESpace)
    n = fe.n
    m = fe.m
    u = zeros(n)
    b = zeros(m)
    F = fe.up(f)
    G = fe.up(g)
    λ = zeros(n)
    δσ = zeros(n)
    rhs = zeros(n)
    return FerriteEITMode(u, nothing, nothing, b, λ, δσ, F, f, G, g, rhs, 0.0, 0.0, 0.0)
end


function state_adjoint_step_dirichlet_cg!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    objective_dirichlet_cg!(mode, sol, fe, maxiter)
    gradient_dirichlet_cg!(mode, sol, fe, maxiter)
    return mode.δσ, mode.error_n
end

function objective_dirichlet_cg!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    d = sol.d
    ∂d = sol.∂d
    L = sol.LD
    down = fe.down
    up = fe.up
    # We solve the state equation ∇⋅(σ∇uᵢ) = 0 : u = f
    cg!(mode.u_f, L, mode.F; maxiter=maxiter)
    # Normalize
    mean_boundary!(mode.u_f, mode, down)
    mode.error_d = d(mode.b, mode.g)
    return mode.error_d
end

function gradient_dirichlet_cg!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    d = sol.d
    ∂d = sol.∂d
    L = sol.LD
    down = fe.down
    up = fe.up
    # We solve the adjoint equation ∇⋅(σ∇λᵢ) = 0 : σ∂λ/∂𝐧 = ∂ₓd(u,f)
    cg!(mode.λ, L, up(∂d(mode.b, mode.g)); maxiter=maxiter)
    # Calculate ∂J(σ,f,g)/∂σ = ∇(uᵢ)⋅∇(λᵢ) here:
    mode.δσ = calculate_bilinear_map!(fe, mode.rhs, mode.λ, mode.u_g)
    return mode.δσ
end
