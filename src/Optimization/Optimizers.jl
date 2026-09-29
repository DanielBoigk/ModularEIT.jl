# Optimisation layer: minimize(obj, σ₀, method) for any AbstractObjective with coefficient
# gradients, optionally with box constraints lower ≤ σ ≤ upper.
#
# Box constraints follow the two-metric projection idea (Bertsekas 1982): variables at a bound
# whose gradient pushes them outward form the binding set 𝒜 and are frozen; the direction is
# computed on the free set ℱ (Riesz map / quasi-Newton / Gauss–Newton restricted to ℱ) and the
# step is taken along the projection arc σ(t) = P(σ - t d) with an Armijo condition on the actual
# (projected) change. For an SPD Riesz map R the restricted direction d = (R g_ℱ)|_ℱ is a descent
# direction, gᵀd = g_ℱᵀ R g_ℱ > 0, also for non-diagonal R such as the inverse σ mass matrix.
#
# Optimality is measured by the projected gradient σ - P(σ - g) (= g without bounds).

"""
    OptimizationState

Result and state of [`minimize`](@ref): `σ` (current iterate), `g` (coefficient gradient),
`value`, `iteration`, `nevals` (objective evaluations), `status` (`:gtol`, `:ftol`,
`:ftarget`, `:maxiter`, `:callback`, `:linesearch`, `:running`), `converged`, and `history`
(one entry `(value, gnorm, step, nevals)` per iterate, starting with σ₀; `gnorm` is the norm of
the projected gradient, `step` the norm of the last change of σ).
"""
mutable struct OptimizationState <: AbstractSolutionState
    σ::Vector{Float64}
    g::Vector{Float64}
    value::Float64
    iteration::Int
    nevals::Int
    status::Symbol
    converged::Bool
    history::Vector{@NamedTuple{value::Float64, gnorm::Float64, step::Float64, nevals::Int}}
end

# lower/upper bounds as vectors (±Inf where unbounded)
struct _Box
    lower::Vector{Float64}
    upper::Vector{Float64}
    active::Bool
end

function _make_box(lower, upper, n)
    expand(b, default) = b === nothing ? fill(default, n) : b isa Real ? fill(Float64(b), n) : Vector{Float64}(b)
    lo, hi = expand(lower, -Inf), expand(upper, Inf)
    length(lo) == n && length(hi) == n || throw(DimensionMismatch("bounds need $n entries"))
    all(lo .<= hi) || throw(ArgumentError("lower bounds must not exceed upper bounds"))
    return _Box(lo, hi, lower !== nothing || upper !== nothing)
end

_clamp_box!(σ, box::_Box) = (box.active && (σ .= clamp.(σ, box.lower, box.upper)); σ)

# ‖σ - P(σ - g)‖
function _projected_gradient_norm(σ, g, box::_Box)
    box.active || return norm(g)
    s = 0.0
    @inbounds for i in eachindex(σ)
        s += (σ[i] - clamp(σ[i] - g[i], box.lower[i], box.upper[i]))^2
    end
    return sqrt(s)
end

# binding set: at (within ε of) a bound with the gradient pushing outward
function _binding!(mask::BitVector, σ, g, box::_Box, ε)
    fill!(mask, false)
    box.active || return mask
    @inbounds for i in eachindex(σ)
        mask[i] = (σ[i] <= box.lower[i] + ε && g[i] > 0) || (σ[i] >= box.upper[i] - ε && g[i] < 0)
    end
    return mask
end
_binding_tolerance(σ, gnorm) = min(gnorm, sqrt(eps()) * (1 + norm(σ, Inf)))

# evaluation that treats a failed factorisation (e.g. a trial σ with nonpositive entries) as +Inf
function _try_value_and_gradient!(g, obj, σ)
    try
        return value_and_gradient!(g, obj, σ)
    catch e
        e isa _INFEASIBLE_EXCEPTIONS || rethrow()
        return Inf
    end
end

"""
    minimize(obj, σ₀, method; lower = nothing, upper = nothing, maxiter = 100, gtol = 1e-6,
             ftol = 0, ftarget = -Inf, callback = nothing, verbose = false)

Minimize the objective `obj` starting from `σ₀` with `method` ([`GradientDescent`](@ref),
[`LBFGS`](@ref), [`GaussNewton`](@ref)) subject to `lower ≤ σ ≤ upper` (scalars or vectors;
`nothing` = unbounded). Returns an [`OptimizationState`](@ref).

Stopping criteria:
- `gtol`: projected gradient norm ≤ `gtol` × its initial value;
- `ftol`: relative decrease of the objective in one iteration ≤ `ftol`;
- `ftarget`: objective ≤ `ftarget`, e.g. the discrepancy principle `J ≤ τ² δ²/2` for a
  (whitened) noise level `δ` and `τ` slightly above 1;
- `maxiter` iterations; `callback(state)` returning `true`.

The objective must deliver coefficient gradients; the Riesz map (gradient representation) is an
option of the method.
"""
function minimize(obj::AbstractObjective, σ0::AbstractVector, method::AbstractOptimizer;
                  lower = nothing, upper = nothing, maxiter::Integer = 100, gtol::Real = 1e-6,
                  ftol::Real = 0.0, ftarget::Real = -Inf, callback = nothing, verbose::Bool = false)
    _require_coefficient_gradient(obj)
    n = length(σ0)
    box = _make_box(lower, upper, n)
    σ = _clamp_box!(Vector{Float64}(σ0), box)
    ws = _workspace(method, obj, n)
    st = OptimizationState(σ, zeros(n), NaN, 0, 0, :running, false, [])
    _initialize!(st, ws, method, obj, box)
    gnorm0 = _record!(st, ws, method, box, 0.0)
    verbose && _log(st)
    stop!(Jold) = _check_stop!(st, ws, method, gnorm0, gtol, ftol, ftarget, Jold, maxiter, callback)
    stop!(NaN) && return st
    while true
        Jold = st.value
        σold = copy(st.σ)
        ok = _step!(st, ws, method, obj, box)
        if !ok
            st.status = :linesearch
            st.converged = _gtol_met(st, ws, method, _optimality(st, ws, method, box), gnorm0, gtol)
            return st
        end
        st.iteration += 1
        _record!(st, ws, method, box, norm(st.σ .- σold))
        verbose && _log(st)
        stop!(Jold) && return st
    end
end

# optimality measure recorded as `gnorm` and the convergence test on it; methods may override
_optimality(st, ws, method, box) = _projected_gradient_norm(st.σ, st.g, box)
_gtol_met(st, ws, method, gnorm, gnorm0, gtol) = gnorm <= gtol * gnorm0 || gnorm == 0

function _record!(st::OptimizationState, ws, method, box, step)
    gnorm = _optimality(st, ws, method, box)
    push!(st.history, (value = st.value, gnorm, step, nevals = st.nevals))
    return gnorm
end

_log(st) = (h = st.history[end]; @info "iteration $(st.iteration)" h.value h.gnorm h.step h.nevals)

function _check_stop!(st, ws, method, gnorm0, gtol, ftol, ftarget, Jold, maxiter, callback)
    gnorm = st.history[end].gnorm
    status = _gtol_met(st, ws, method, gnorm, gnorm0, gtol) ? :gtol :
             st.value <= ftarget ? :ftarget :
             ftol > 0 && isfinite(Jold) && Jold - st.value <= ftol * max(abs(Jold), floatmin()) ? :ftol :
             callback !== nothing && callback(st) === true ? :callback :
             st.iteration >= maxiter ? :maxiter : :running
    status === :running && return false
    st.status = status
    st.converged = status in (:gtol, :ftarget, :ftol)
    return true
end

# default: first evaluation with gradient
function _initialize!(st, ws, method, obj, box)
    st.value = value_and_gradient!(st.g, obj, st.σ)
    st.nevals += 1
    isfinite(st.value) || throw(ArgumentError("the objective is not finite at the initial guess"))
    return st
end

# Backtracking along the projection arc σ(t) = P(σ - t d), Armijo: J(σ(t)) ≤ J - c gᵀ(σ - σ(t)).
# On success the state holds σ(t), its value and gradient; `σt`, `gt` are buffers.
function _projected_backtracking!(st, obj, box, d, t, σt, gt; c = 1e-4, shrink = 0.5, maxtries = 60)
    for _ in 1:maxtries
        σt .= st.σ .- t .* d
        _clamp_box!(σt, box)
        decrease = sum(i -> st.g[i] * (st.σ[i] - σt[i]), eachindex(σt))
        if decrease > 0
            Jt = _try_value_and_gradient!(gt, obj, σt)
            st.nevals += 1
            if Jt <= st.value - c * decrease
                copyto!(st.σ, σt)
                copyto!(st.g, gt)
                st.value = Jt
                return t
            end
        elseif st.σ == σt
            return 0.0              # nothing moves any more
        end
        t *= shrink
    end
    return 0.0
end

# initial step so that the largest coefficient changes by 10 % of the scale of σ
_initial_step(σ, d) = (m = norm(d, Inf); m > 0 ? 0.1 * max(norm(σ, Inf), 1.0) / m : 1.0)

# d ← (R g_ℱ)|_ℱ
function _restricted_riesz!(d, R, g, mask)
    d .= g
    d[mask] .= 0
    riesz_map!(d, R)
    d[mask] .= 0
    return d
end
