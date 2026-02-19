export create_f∂f

"""
create_f∂f(prblm, num_modes::Int=100; regularize::Bool=false, gn::Bool=false)

Create objective (`f`) and gradient (`g`) closures for EIT reconstruction.

Arguments
- `prblm` : Problem object that contains model `state`, `modes`, solvers and options used by the closures.
- `num_modes::Int=100` : Number of modes to solve and accumulate into the objective and gradient.
- `regularize::Bool=false` : If `true`, add diffusion/Tikhonov-style regularization terms to both objective and gradient.
- `gn::Bool=false` : Whether Gauss–Newton is used. This flag is present for API/future use but is ignored by the current implementation.

Note
- The parameter `ph` (pointhandler) used to be accepted by this function but is no longer required by the implementation and has been removed from the API.

Returns
- `(f, g)` : A tuple of functions
    - `f(σ)` : Given a conductivity vector/array `σ`, returns the scalar objective (misfit) for that `σ`.
    - `g(σ)` : Given a conductivity vector/array `σ`, returns the gradient of the objective w.r.t. `σ`.

Behavior and notes
- Both closures enforce a positive lower bound on conductivity via `σc = max.(σ, 1e-6)` (a clamped copy), avoiding mutation of the caller's array before cache checks.
- `f` reuses cached results stored in `prblm.state` when the clamped `σc` exactly equals `prblm.state.σ` to avoid recomputation.
- `g` reuses a cached gradient if `prblm.state.δ_updated` is true and the state hasn't changed.
- When the state is updated the closures:
    1. copy the clamped `σc` into `prblm.state.σ`,
    2. call `update_L!(prblm.state, prblm.fe, true)`,
    3. call `solve_modes!` to compute per-mode contributions,
    4. accumulate per-mode misfits `prblm.modes[i].error_n` into `prblm.state.error`,
    5. for `g`, accumulate per-mode gradient contributions `prblm.modes[i].δσ` into `prblm.state.δ`.
- If `regularize` is true:
    - `f` adds `prblm.state.opt.β_diff * prblm.state.R_diff(prblm.state.σ)` to the objective.
    - `g` adds `prblm.state.opt.β_diff * prblm.state.∇R(prblm.state.σ)` to the gradient.
- Side effects: The closures mutate fields on `prblm.state` (e.g. `σ`, `error`, `δ`, `δ_updated`) and call mutating helper functions. They are not pure functions and require exclusive access to `prblm` for thread-safety.

Example
    (f, g) = create_fg(prblm, 50; regularize=true, gn=true)
    J = f(σ0)
    ∇J = g(σ0)

Note: The `gn` (Gauss–Newton) flag is included in the signature for future behavior changes but is ignored by the current code.

"""
function create_f∂f(prblm, num_modes::Int=100; regularize::Bool=false, gn::Bool=false)
    f = σ -> begin
        σc = max.(σ, 1e-6)
        if σc == prblm.state.σ
            return prblm.state.error
        end
        prblm.state.σ .= σc
        update_L!(prblm.state, prblm.fe, true)
        solve_modes!(prblm, num_modes, objective_neumann_init!)
        prblm.state.error = sum(prblm.modes[i].error_n for i in 1:num_modes)
        if regularize
            prblm.state.error += prblm.state.opt.β_diff * prblm.state.R_diff(prblm.state.σ)
        end
        prblm.state.δ_updated = false
        return prblm.state.error
    end
    ∂f = σ -> begin
        σc = max.(σ, 1e-6)
        if σc != prblm.state.σ
            prblm.state.σ .= σc
            update_L!(prblm.state, prblm.fe, true)
            solve_modes!(prblm, num_modes, objective_neumann_init!)
            prblm.state.error = sum(prblm.modes[i].error_n for i in 1:num_modes)
            if regularize
                prblm.state.error += prblm.state.opt.β_diff * prblm.state.R_diff(prblm.state.σ)
            end
        elseif prblm.state.δ_updated
            return copy(prblm.state.δ)
        end
        prblm.state.δ_updated = true
        fill!(prblm.state.δ, 0.0)
        solve_modes!(prblm, num_modes, gradient_neumann_init!)
        for i in 1:num_modes
            prblm.state.δ .-= prblm.modes[i].δσ
        end
        if regularize
            prblm.state.δ .+= (prblm.state.opt.β_diff * prblm.state.∇R(prblm.state.σ))
        end
        return copy(prblm.state.δ)
    end
    return f, ∂f
end
