using LinearMaps, IterativeSolvers
export create_f∂f, create_prox_linesearch, create_proximal_gradient_step

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
function create_f∂f(prblm, num_modes::Int=100; gn::Bool=false, mode="neumann", obj=objective_neumann_init!, grad=gradient_neumann_init!, gauss_newton=gauss_newton_svd!, return_grads::Bool = false, normalize = true)
    # Flag to force computation on first call to avoid returning sentinel value
    first_call = Ref(true)

    f = σ -> begin
        σc = max.(σ, 1e-6)
        if !first_call[] && σc == prblm.state.σ
            # Cache hit (but not on first call)
            return prblm.state.error
        end
        first_call[] = false

        # Cache miss or first call - recompute
        prblm.state.σ .= σc
        update_L!(prblm.state, prblm.fe, true, mode ≠ "neumann")
        solve_modes!(prblm, num_modes, obj)

        collect_r!(prblm, num_modes, mode=mode)
        if normalize
            prblm.state.error = sum(prblm.state.opt.r) / (num_modes*prblm.fe.n^2)
        else
            prblm.state.error = sum(prblm.state.opt.r)
        end
        prblm.state.δ_updated = false
        return prblm.state.error
    end
    if return_grads
        ∂f = σ -> begin
            σc = max.(σ, 1e-6)
            if σc != prblm.state.σ
                f(σc)
            elseif prblm.state.δ_updated
                return copy(prblm.state.δ)
            end
            prblm.state.δ_updated = true
            fill!(prblm.state.δ, 0.0)
            solve_modes!(prblm, num_modes, grad)
            collect_J!(prblm, num_modes)

            return copy(prblm.state.opt.J)
        end
    else
        ∂f = σ -> begin
            σc = max.(σ, 1e-6)
            if σc != prblm.state.σ
                f(σc)
            elseif prblm.state.δ_updated
                return copy(prblm.state.δ)
            end
            prblm.state.δ_updated = true
            fill!(prblm.state.δ, 0.0)
            solve_modes!(prblm, num_modes, grad)
            collect_J!(prblm, num_modes)
            if gn
                gauss_newton(prblm.state.opt)
                prblm.state.δ = copy(prblm.state.opt.δ)
            else
                # for i in 1:num_modes
                #     prblm.state.δ .-= prblm.modes[i].δσ
                # end
                prblm.state.δ = vec(sum(prblm.state.opt.J; dims=1))
            end
            return copy(prblm.state.δ)
        end
    end
    return f, ∂f
end

using Optim

function create_prox_linesearch(f, ∂f, ρ::Number =0.0)
    ρ_internal = ρ
    prox = (x, ρ = ρ_internal) -> begin
        current_val = f(x)
        direction = ∂f(x)
        τ_min, τ_max = determine_box(x, direction)
        
        # Brent's method in Optim.jl takes a pure scalar function
        opt_func = (τ) -> f(x + τ * direction) + 0.5 * ρ * sum(abs2, τ*direction)
        
        # optimize(f, lower, upper, method)
        # Note: Brent() is the default for this syntax in Optim
        results = optimize(opt_func, τ_min, τ_max, Brent())
        
        # Extract the scalar minimizer and the minimum value
        best_τ = Optim.minimizer(results)
        new_val = Optim.minimum(results)
        
        if new_val < current_val
            return x + best_τ * direction, new_val, 0.5 * ρ * sum(abs2, best_τ*direction)
        end
        return x, current_val, 0.0
    end
    return prox
end

# Todo: Fix this function rewrite it as a Line search:
export create_proximal_gradient_step
function create_proximal_gradient_step(f, ∂f, ρ;λ::Number=1.0, kwargs... )
    prox = v -> begin
        if λ != 1.0
            objective = x -> λ * f(x) + 0.5 * ρ * sum(abs2, x .- v)
            gradient = x -> λ .* ∂f(x) .+ ρ .* (x .- v)
        else
            objective = x -> f(x) + 0.5 * ρ * sum(abs2, x .- v)
            gradient = x -> ∂f(x) .+ ρ .* (x .- v)
        end
        # This is my own private version of box constrained lbfgs:
        result = lbfgs_b(objective,gradient, v; kwargs...)
        return result
    end
    return prox
end

export create_tikhonov
function create_tikhonov(K::AbstractArray;β::Float64=1.0, ρ::Float64 = 1.0)
    R = (x) ->  (β/2) * dot(x, K*x)
    ∇R = (x) -> β * K*x
    prox = (x) -> begin
        A = β*K+ρ*I # This should be LinearMap
        b = ρ*x
        out = A \ b
        return out,R(x), ρ*sum(abs2, x .- out)
    end

    return R, ∇R, prox
end

export create_totalvariation
function create_totalvariation(fe::FerriteFESpace,β::Float64=1.0; ρ::Float64 = 1.0, ϵ::Float64=1e-6)
    TV = (x) -> (β/2) * normTV(fe, x)
    ∇TV = (x) -> β * assemble_huber_gradient(fe, x, ϵ)
    prox_TV = (v) -> begin
        # 1. Define the objective specifically for the target vector `v`
        obj = (x) -> TV(x) + 0.5 * ρ * sum(abs2, x .- v)
        
        # 2. Define an in-place gradient function `g!(storage, x)` for Optim
        grad! = (storage, x) -> begin
            storage .= ∇TV(x) .+ ρ .* (x .- v)
        end
        
        # 3. Optimize using LBFGS with the in-place gradient
        result = optimize(obj, grad!, copy(v), LBFGS())
        xmin = Optim.minimizer(result)
        
        return xmin, TV(xmin), 0.5 * ρ * sum(abs2, v .- xmin)
    end
    return TV,∇TV,prox_TV
end