using LinearAlgebra
using IterativeSolvers

export FerriteEITModeN
export state_adjoint_step_dirichlet_cg!
export objective_dirichlet_cg!
export gradient_dirichlet_cg!

"""
    FerriteEITModeD(g::AbstractVector, f::AbstractVector, fe::FerriteFESpace)

Constructs a Dirichlet-mode configuration for an Electrical Impedance Tomography (EIT) problem
using a given finite element space `fe`. Initializes all relevant vectors and parameters
for the state, adjoint, and gradient computation.

# Arguments
- `g`: Boundary measurement vector.
- `f`: Boundary excitation vector.
- `fe`: Finite element space (FerriteFESpace) describing mesh, degrees of freedom, and lifting operators.

# Returns
`FerriteEITMode` struct initialized for Dirichlet boundary problems.
"""
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


"""
    objective_dirichlet_cg!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace; maxiter=500)

Solve the forward (state) problem of EIT with Dirichlet boundary conditions using a conjugate gradient solver.
The PDE ∇⋅(σ∇uᵢ) = 0 is solved with `u|∂Ω = f`.

# Arguments
- `mode`: Current EIT mode state.
- `sol`: Solver state containing the differential operator `L = LD` and data misfit operators `d`, `∂d`.
- `fe`: Finite element space defining up/down maps.
- `maxiter`: Maximum number of CG iterations.

# Effects
- Updates `mode.u_f` with the computed potential field.
- Normalizes boundary potentials.
- Updates `mode.error_d` with the current data misfit value.

# Returns
`mode.error_d` — the current Dirichlet objective (data misfit).
"""
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

"""
    gradient_dirichlet_cg!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace; maxiter=500)

Solve the adjoint problem of the EIT inverse formulation with Dirichlet boundary conditions.
Computes the gradient of the objective with respect to the conductivity σ.

# Equations
Solves ∇⋅(σ∇λᵢ) = 0 with Neumann-type boundary condition σ∂λ/∂n = ∂ₓd(u,f).
Then computes ∂J/∂σ = ∇u ⋅ ∇λ elementwise.

# Arguments
- `mode`: Current EIT mode state.
- `sol`: Solver state containing the differential operator `L` and loss derivatives `∂d`.
- `fe`: Finite element space defining basis and lifting operators.
- `maxiter`: Maximum number of CG iterations.

# Effects
- Updates `mode.λ` with the adjoint solution.
- Updates `mode.δσ` with the conductivity gradient.

# Returns
`mode.δσ` — the computed gradient with respect to σ.
"""
function gradient_dirichlet_cg!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    d = sol.d
    ∂d = sol.∂d
    L = sol.LD
    down = fe.down
    up = fe.up

    rhs = up(∂d(mode.b, mode.g)
    mean_boundary!(rhs, mode, down)
    # We solve the adjoint equation ∇⋅(σ∇λᵢ) = 0 : σ∂λ/∂𝐧 = ∂ₓd(u,f)
    cg!(mode.λ, L, rhs; maxiter=maxiter)
    # Calculate ∂J(σ,f,g)/∂σ = ∇(uᵢ)⋅∇(λᵢ) here:
    mode.δσ = calculate_bilinear_map!(fe, mode.rhs, mode.λ, mode.u_g)
    return mode.δσ
end

"""
    state_adjoint_step_dirichlet_cg!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace; maxiter=500)

Performs a full forward–adjoint step for the Dirichlet formulation of EIT using conjugate gradients.
Sequentially computes the state (forward) solution, then the adjoint solution, and the resulting
conductivity gradient.

# Arguments
- `mode`: Current EIT mode state.
- `sol`: Solver state containing operators and loss definitions.
- `fe`: Finite element space used for discretization.
- `maxiter`: Maximum number of CG iterations.

# Returns
Tuple `(δσ, error_d)` — the conductivity gradient and the data misfit value.
"""
function state_adjoint_step_dirichlet_cg!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    objective_dirichlet_cg!(mode, sol, fe, maxiter)
    gradient_dirichlet_cg!(mode, sol, fe, maxiter)
    return mode.δσ, mode.error_n
end
