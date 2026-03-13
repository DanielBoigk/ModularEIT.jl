using IterativeSolvers
using LinearAlgebra
using Ferrite

export FerriteEITModeM
export state_adjoint_step_mixed_cg!
export objective_mixed_cg!
export gradient_mixed_cg!
export objective_mixed_init
export gradient_mixed_init
export state_adjoint_step_mixed_init!


"""
    FerriteEITModeM(g::AbstractVector, f::AbstractVector, fe::FerriteFESpace)

Constructs a mixed-mode configuration for an Electrical Impedance Tomography (EIT) problem
combining both Dirichlet and Neumann boundary data.

# Arguments
- `g`: Boundary excitation (current) vector.
- `f`: Boundary potential (voltage) vector.
- `fe`: Finite element space (`FerriteFESpace`) describing mesh, degrees of freedom, and lifting operators.

# Returns
`FerriteEITMode` struct initialized for mixed boundary problems.
"""
function FerriteEITModeM(g::AbstractVector, f::AbstractVector, fe::FerriteFESpace)
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
    λrhs = zeros(n)
    rhs = zeros(n)
    return FerriteEITMode(u_f, u_g, w, b, λ, δσ, F, f, G, g, λrhs, rhs, 0.0, 0.0, 0.0)
end

"""
    state_adjoint_step_mixed_cg!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace; maxiter=500)

Performs a complete forward–adjoint step for the mixed boundary formulation of the EIT problem.
Sequentially computes:
1. The forward Dirichlet and Neumann state solutions,
2. The mixed potential field `w = u_f - u_g`,
3. The adjoint solution and conductivity gradient.

# Arguments
- `mode`: Current EIT mode state.
- `sol`: Solver state containing all operators and loss definitions.
- `fe`: Finite element space used for discretization.
- `maxiter`: Maximum number of CG iterations.

# Returns
Tuple `(δσ, error_m)` — the conductivity gradient and the mixed data misfit value.
"""
function state_adjoint_step_mixed_cg!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    objective_mixed_cg!(mode, sol, fe, maxiter)
    gradient_mixed_cg!(mode, sol, fe, maxiter)
    return mode.δσ, mode.error_m
end
function state_adjoint_step_mixed_init!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    objective_mixed_init!(mode, sol, fe, maxiter)
    gradient_mixed_init!(mode, sol, fe, maxiter)
    return mode.δσ, mode.error_m
end

"""
    objective_mixed_cg!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace; maxiter=500)

Solve the mixed (state) problem of EIT combining Dirichlet and Neumann boundary data
using conjugate gradient solvers for both formulations.

# Description
Solves:
- ∇⋅(σ∇u_f) = 0 with `u|∂Ω = f` (Dirichlet)
- ∇⋅(σ∇u_g) = 0 with `σ∂u/∂n = g` (Neumann)

Then constructs the mixed potential field
`w = u_f - u_g`, normalized to zero mean,
and evaluates the mixed data discrepancy via `n(w)`.

# Arguments
- `mode`: Current EIT mode state.
- `sol`: Solver state containing system operators and data misfit functions.
- `fe`: Finite element space defining lifting maps and mesh geometry.
- `maxiter`: Maximum number of CG iterations.

# Effects
- Updates `mode.u_f`, `mode.u_g`, and `mode.w`.
- Updates `mode.error_m` with the mixed data misfit.

# Returns
`mode.error_m` — the current mixed objective value.
"""
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

function objective_mixed_init!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace)
    n = sol.n
    # We solve the state equation ∇⋅(σ∇uᵢ) = 0 : u = f
    objective_dirichlet_init!(mode, sol, fe)
    # We solve the state equation ∇⋅(σ∇uᵢ) = 0 : σ∂u/∂𝐧 = g
    objective_neumann_init!(mode, sol, fe)

    mode.w = mode.u_f - mode.u_g
    mode.error_m = n(mode.w)
    return mode.error_m
end


"""
    gradient_mixed_cg!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace; maxiter=500)

Solve the adjoint problem for the mixed boundary formulation of EIT and compute the
gradient of the mixed objective with respect to the conductivity σ.

# Equations
Solves ∇⋅(σ∇λ) = ∂ₙ(w)
and computes the gradient elementwise as ∂J/∂σ = ∇w ⋅ ∇λ.

# Arguments
- `mode`: Current EIT mode state.
- `sol`: Solver state containing differential operator `L` and derivative operator `∂n`.
- `fe`: Finite element space defining the basis and assembly operations.
- `maxiter`: Maximum number of CG iterations.

# Effects
- Updates `mode.λ` with the adjoint solution.
- Updates `mode.δσ` with the conductivity gradient.

# Returns
`mode.δσ` — the computed mixed-mode gradient with respect to σ.
"""
function gradient_mixed_cg!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    Ln = sol.L
    ∂n = sol.∂n
    # This one needs to normalize:
    mode.λrhs = ∂n(mode.w)
    mode.λrhs -= mean(mode.λrhs)
    # We solve the adjoint equation ∇⋅(σ∇λᵢ) = ∂n(w)
    cg!(mode.λ, Ln, mode.λrhs; maxiter=maxiter)
    # Calculate ∇(uᵢ)⋅∇(λᵢ) here:
    mode.δσ = calculate_bilinear_map!(fe, mode.rhs, mode.λ, mode.w)
    return mode.δσ
end

function gradient_mixed_init!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace)
    L = sol.L_fac
    ∂n = sol.∂n
    # This one needs to normalize:
    mode.λrhs = ∂n(mode.w)
    mode.λrhs -= mean(mode.λrhs)
    # We solve the adjoint equation ∇⋅(σ∇λᵢ) = ∂n(w)
    mode.λ = L \ mode.λrhs
    # Calculate ∇(uᵢ)⋅∇(λᵢ) here:
    mode.δσ = calculate_bilinear_map!(fe, mode.rhs, mode.λ, mode.w)
    return mode.δσ
end
