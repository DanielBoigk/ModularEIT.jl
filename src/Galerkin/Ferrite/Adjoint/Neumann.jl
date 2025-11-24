using LinearAlgebra
using IterativeSolvers

export FerriteEITModeN
export state_adjoint_step_neumann_cg!
export objective_neumann_cg!
export gradient_neumann_cg!

"""
    FerriteEITModeN(g::AbstractVector, f::AbstractVector, fe::FerriteFESpace)

Constructs a Neumann-mode configuration for an Electrical Impedance Tomography (EIT) problem
using a given finite element space `fe`. Initializes all relevant vectors and parameters
for the state, adjoint, and gradient computations.

# Arguments
- `g`: Boundary excitation (current) vector.
- `f`: Boundary measurement (potential) vector.
- `fe`: Finite element space (`FerriteFESpace`) describing mesh, degrees of freedom, and lifting operators.

# Returns
`FerriteEITMode` struct initialized for Neumann boundary problems.
"""
function FerriteEITModeN(g::AbstractVector, f::AbstractVector, fe::FerriteFESpace)
    n = fe.n
    m = fe.m
    u = zeros(n)
    b = zeros(m)
    F = fe.up(f)
    G = fe.up(g)
    λ = zeros(n)
    λrhs = zeros(n)
    δσ = zeros(n)
    rhs = zeros(n)
    return FerriteEITMode(nothing, u, nothing, b, λ, δσ, F, f, G, g, λrhs, rhs, 0.0, 0.0, 0.0)
end


"""
    state_adjoint_step_neumann_cg!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace; maxiter=500)

Performs a full forward–adjoint step for the Neumann formulation of the EIT problem
using conjugate gradients. Sequentially computes the state (forward) solution,
then the adjoint solution, and the resulting conductivity gradient.

# Arguments
- `mode`: Current EIT mode state.
- `sol`: Solver state containing system operators and loss definitions.
- `fe`: Finite element space used for discretization.
- `maxiter`: Maximum number of CG iterations.

# Returns
Tuple `(δσ, error_n)` — the conductivity gradient and the Neumann data misfit value.
"""
function state_adjoint_step_neumann_cg!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    objective_neumann_cg!(mode, sol, fe, maxiter)
    gradient_neumann_cg!(mode, sol, fe, maxiter)
    return mode.δσ, mode.error_n
end

function state_adjoint_step_neumann_init!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    objective_neumann_init!(mode, sol, fe, maxiter)
    gradient_neumann_init!(mode, sol, fe, maxiter)
    return mode.δσ, mode.error_n
end

"""
    objective_neumann_cg!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace; maxiter=500)

Solve the forward (state) problem of EIT with Neumann boundary conditions using a conjugate gradient solver.
The PDE ∇⋅(σ∇uᵢ) = 0 is solved with boundary condition σ∂u/∂n = g.

# Arguments
- `mode`: Current EIT mode state.
- `sol`: Solver state containing the operator `L` and data misfit function `d`.
- `fe`: Finite element space defining up/down lifting maps.
- `maxiter`: Maximum number of CG iterations.

# Effects
- Updates `mode.u_g` with the computed potential field.
- Normalizes the boundary potential using `mean_boundary!`.
- Updates `mode.error_n` with the current data misfit.

# Returns
`mode.error_n` — the current Neumann objective (data misfit).
"""
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

function objective_neumann_init!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    d = sol.d
    ∂d = sol.∂d
    L = sol.L_fac
    down = fe.down
    up = fe.up
    # We solve the state equation ∇⋅(σ∇uᵢ) = 0 : σ∂u/∂𝐧 = g
    mode.u_g .= L \ mode.G
    # Normalize
    mean_boundary!(mode.u_g, mode, down)
    mode.error_n = d(mode.b, mode.f)
    return mode.error_n
end


"""
    gradient_neumann_cg!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace; maxiter=500)

Solve the adjoint problem for the Neumann formulation of EIT and compute the gradient
of the objective with respect to the conductivity σ.

# Equations
Solves ∇⋅(σ∇λᵢ) = 0 with boundary condition σ∂λ/∂n = ∂ₓd(u,f),
then computes ∂J/∂σ = ∇u ⋅ ∇λ elementwise.

# Arguments
- `mode`: Current EIT mode state.
- `sol`: Solver state containing differential operator `L` and loss derivative `∂d`.
- `fe`: Finite element space defining basis and lifting operators.
- `maxiter`: Maximum number of CG iterations.

# Effects
- Updates `mode.λ` with the adjoint field.
- Updates `mode.δσ` with the conductivity gradient.

# Returns
`mode.δσ` — the computed gradient with respect to σ.
"""
function gradient_neumann_cg!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    d = sol.d
    ∂d = sol.∂d
    L = sol.L
    down = fe.down
    up = fe.up
    mode.λrhs = up(∂d(mode.b, mode.f))
    mean_boundary!(mode.λrhs, mode, down)
    # We solve the adjoint equation ∇⋅(σ∇λᵢ) = 0 : σ∂λ/∂𝐧 = ∂ₓd(u,f)
    cg!(mode.λ, L, mode.λrhs; maxiter=maxiter)
    # Calculate ∂J(σ,f,g)/∂σ = ∇(uᵢ)⋅∇(λᵢ) here:
    mode.δσ = calculate_bilinear_map!(fe, mode.rhs, mode.λ, mode.u_g)
    return mode.δσ
end


function gradient_neumann_init!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    d = sol.d
    ∂d = sol.∂d
    L = sol.L_fac
    down = fe.down
    up = fe.up
    mode.λrhs = up(∂d(mode.b, mode.f))
    mean_boundary!(mode.λrhs, mode, down)
    # We solve the adjoint equation ∇⋅(σ∇λᵢ) = 0 : σ∂λ/∂𝐧 = ∂ₓd(u,f)
    mode.λ = L \ mode.λrhs
    # Calculate ∂J(σ,f,g)/∂σ = ∇(uᵢ)⋅∇(λᵢ) here:
    mode.δσ = calculate_bilinear_map!(fe, mode.rhs, mode.λ, mode.u_g)
    return mode.δσ
end

using Base.Threads

"""
    solve_all_neumann_cg!(modes::Dict{Int,FerriteEITMode}, sol::FerriteSolverState, fe::FerriteFESpace; maxiter=500)

Runs the Neumann EIT state–adjoint solver for all entries in `modes` concurrently using multithreading.

# Arguments
- `modes`: Dict mapping mode indices to `FerriteEITMode` objects.
- `sol`: Shared `FerriteSolverState` containing operators and loss definitions.
- `fe`: Finite element space.
- `maxiter`: Maximum CG iterations.

# Returns
A `Dict{Int,Tuple{Vector{Float64},Float64}}` mapping each mode index to `(δσ, error_n)`.
"""
function solve_all_neumann_cg!(modes::Dict{Int,FerriteEITMode}, sol::FerriteSolverState, fe::FerriteFESpace; maxiter=500)
    keys_vec = collect(keys(modes))
    results = Dict{Int,Tuple{Vector{Float64},Float64}}()
    lock = ReentrantLock()

    @threads for k in keys_vec
        mode = modes[k]
        δσ, err = state_adjoint_step_neumann_cg!(mode, sol, fe, maxiter)
        lock(lock) do
            results[k] = (δσ, err)
        end
    end

    return results
end
