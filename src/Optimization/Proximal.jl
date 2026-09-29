# Proximal methods for J(σ) = F(σ) + α G(σ): F smooth (the objective passed to minimize, e.g. a
# data misfit, possibly a RegularizedObjective), G a regularizer used only through its proximal
# operator (non-smooth TV, bounds, a denoiser). Both methods work in a diagonal metric w
# (Euclidean or the lumped σ mass matrix): gradients are turned into directions by w⁻¹ and the
# proximal operators are taken in ‖·‖_w, so with w = lumped mass the iterations discretise the
# L² versions of the methods and behave alike on all meshes.

_weights(w, n) = w === nothing ? ones(n) : (length(w) == n ? Vector{Float64}(w) :
                                            throw(DimensionMismatch("weights need $n entries")))
_bound_or_nothing(box::_Box, b) = box.active ? b : nothing

# z = argmin α G(z) + ρ/2 ‖z - v‖²_w over the box. Iterative proximal maps are solved inexactly:
# the error ‖z - z*‖_w only has to be small compared with the method's current progress `scale`
# (ADMM: primal residual, proximal gradient: last step), which the gap tolerance
# (ρ/α)/2 (0.1 scale)² guarantees. Inexact ADMM and proximal gradient methods converge when these
# errors decrease with the progress; solving every prox to round-off would waste most of the time.
function _scaled_prox!(z, α, G, v, ρ, w, box; scale = nothing)
    if α == 0
        copyto!(z, v)
        return _clamp_box!(z, box)
    end
    ρG = ρ / α
    tol, maxiter = scale === nothing || !isfinite(scale) ? (nothing, nothing) : (ρG / 2 * (0.1scale)^2, 5000)
    return prox!(z, G, v, ρG; weights = w, lower = _bound_or_nothing(box, box.lower),
                 upper = _bound_or_nothing(box, box.upper), tol, maxiter)
end

# ---------------------------------------------------------------------------------------------
# Proximal gradient (forward–backward splitting, FISTA)
#
#     z = prox_{tαG}(y - t w⁻¹ ∇F(y))  in ‖·‖_w,
#
# step t by backtracking on F(z) ≤ F(y) + ∇F(y)ᵀ(z - y) + ‖z - y‖²_w / 2t (and a mild increase
# t ← 1.25 t after every iteration). Acceleration: monotone FISTA (Beck & Teboulle 2009) with
# restart of the momentum whenever the objective does not decrease. Optimality: the gradient
# mapping w (y - z)/t (= ∇F for G = 0 without bounds).

"""
    ProximalGradient(α => G; weights = nothing, accelerated = true)

Proximal gradient method (FISTA) for `F(σ) + α G(σ)`: `F` is the objective passed to
[`minimize`](@ref), `G` a regularizer with a proximal operator ([`prox!`](@ref)), e.g. the
exact total variation `TotalVariationRegularizer(disc; ε = 0)` or a [`ProximalMap`](@ref).
Bounds are part of the proximal step. `weights`: diagonal metric (default Euclidean; use
[`lumped_mass`](@ref) for the L² metric). `accelerated`: monotone FISTA with restarts.

The history records `F + αG` and the norm of the gradient mapping.
"""
struct ProximalGradient{R <: AbstractRegularizer, W} <: AbstractOptimizer
    α::Float64
    G::R
    weights::W
    accelerated::Bool
end
function ProximalGradient(term::Pair{<:Real, <:AbstractRegularizer}; weights = nothing, accelerated::Bool = true)
    first(term) >= 0 || throw(ArgumentError("α must be nonnegative"))
    return ProximalGradient(Float64(first(term)), last(term), weights, accelerated)
end

mutable struct _ProxGradWorkspace
    w::Vector{Float64}
    y::Vector{Float64}           # extrapolated point
    xprev::Vector{Float64}
    z::Vector{Float64}
    gy::Vector{Float64}
    v::Vector{Float64}
    t::Float64
    θ::Float64
    step::Float64                # last ‖z - y‖_w: accuracy scale of the inexact prox
end

_workspace(m::ProximalGradient, obj, n) =
    _ProxGradWorkspace(_weights(m.weights, n), zeros(n), zeros(n), zeros(n), zeros(n), zeros(n), NaN, 1.0, NaN)

# accuracy scale for the prox: the last step, or 1 % of the point at the start
_prox_scale(step, w, v) = isfinite(step) && step > 0 ? step : 1e-2 * _wnorm(w, v)

_regularizer_value(α, G, σ) = α == 0 ? 0.0 : α * objective_value(G, σ)

function _initialize!(st, ws::_ProxGradWorkspace, m::ProximalGradient, obj, box)
    F = value_and_gradient!(ws.gy, obj, st.σ)
    st.nevals += 1
    isfinite(F) || throw(ArgumentError("the objective is not finite at the initial guess"))
    st.value = F + _regularizer_value(m.α, m.G, st.σ)
    copyto!(ws.y, st.σ)
    copyto!(ws.xprev, st.σ)
    ws.t = _initial_step(st.σ, ws.gy ./ ws.w)
    # gradient mapping at σ₀ with the initial step
    ws.v .= st.σ .- ws.t .* ws.gy ./ ws.w
    _scaled_prox!(ws.z, m.α, m.G, ws.v, 1 / ws.t, ws.w, box; scale = _prox_scale(ws.step, ws.w, ws.v))
    st.g .= ws.w .* (st.σ .- ws.z) ./ ws.t
    ws.step = _wnorm(ws.w, st.σ .- ws.z)
    return st
end

function _step!(st, ws::_ProxGradWorkspace, m::ProximalGradient, obj, box)
    Fy = _try_value_and_gradient!(ws.gy, obj, ws.y)
    st.nevals += 1
    if !isfinite(Fy)                  # extrapolated point infeasible for the PDE: restart there
        copyto!(ws.y, st.σ)
        ws.θ = 1.0
        Fy = value_and_gradient!(ws.gy, obj, ws.y)
        st.nevals += 1
    end
    Fz = Inf
    for _ in 1:60
        ws.v .= ws.y .- ws.t .* ws.gy ./ ws.w
        _scaled_prox!(ws.z, m.α, m.G, ws.v, 1 / ws.t, ws.w, box; scale = _prox_scale(ws.step, ws.w, ws.v))
        Fz = _try_objective_value(obj, ws.z)
        st.nevals += 1
        d = ws.z .- ws.y
        model = Fy + dot(ws.gy, d) + sum(ws.w .* d .^ 2) / (2ws.t)
        Fz <= model + 1e-12 * abs(Fy) && break
        ws.t /= 2
        Fz = Inf
    end
    isfinite(Fz) || return false
    st.g .= ws.w .* (ws.y .- ws.z) ./ ws.t
    ws.step = _wnorm(ws.w, ws.z .- ws.y)
    Φz = Fz + _regularizer_value(m.α, m.G, ws.z)
    x = st.σ
    copyto!(ws.xprev, x)
    decreased = Φz <= st.value
    if decreased
        copyto!(x, ws.z)
        st.value = Φz
    end
    if m.accelerated && decreased
        θnew = (1 + sqrt(1 + 4ws.θ^2)) / 2
        @. ws.y = x + (ws.θ / θnew) * (ws.z - x) + ((ws.θ - 1) / θnew) * (x - ws.xprev)
        _clamp_box!(ws.y, box)
        ws.θ = θnew
    else
        copyto!(ws.y, x)
        ws.θ = 1.0
    end
    ws.t *= 1.25
    return true
end

# ---------------------------------------------------------------------------------------------
# ADMM for min F(x) + α G(z) s.t. x = z (scaled form, metric w):
#
#     x ← argmin F(x) + ρ/2 ‖x - (z - u)‖²_w        inexact: a few iterations of an inner method
#     z ← argmin α G(z) + ρ/2 ‖z - (x + u)‖²_w       prox (exact for TV, Tikhonov, bounds)
#     u ← u + x - z
#
# The x-step is `minimize` on F plus the proximal term, a TikhonovRegularizer with Gram matrix
# diag(w) whose reference and weight are updated in place, so any method works as the inner
# solver (GaussNewton uses its exact Hessian ρ diag(w)), warm-started at the previous x.
# Residual balancing (Boyd et al. 2011, §3.4.1) adapts ρ; stopping on the primal residual
# ‖x - z‖_w and the dual residual ρ‖z - z_old‖_w relative to ‖x‖_w, ‖z‖_w and ‖ρu‖_w.

"""
    ADMM(α => G; ρ = 1, inner = LBFGS(), inner_maxiter = 10, inner_gtol = 1e-8,
         weights = nothing, adaptive = true)

Alternating direction method of multipliers for `F(σ) + α G(σ)` (see the wiki articles *ADMM*
and *Nested ADMM Reconstruction*): the data step minimizes `F + ρ/2 ‖x - (z - u)‖²_w` with
`inner_maxiter` iterations of the method `inner` (any [`AbstractOptimizer`](@ref), bounds
applied), the regularizer step is the proximal operator of `G` ([`prox!`](@ref), e.g. exact
TV or a [`ProximalMap`](@ref) denoiser for plug-and-play priors). `adaptive`: residual
balancing of `ρ`. Returns the z iterate (feasible for `G` and the bounds).

The history records `F(x) + α G(z)`, the primal residual `‖x - z‖_w` as `gnorm`, and the change
of z as `step`. Convergence (`gtol`): both residuals below `gtol` relative to the iterates and
the scaled dual variable.
"""
struct ADMM{R <: AbstractRegularizer, O <: AbstractOptimizer, W} <: AbstractOptimizer
    α::Float64
    G::R
    ρ::Float64
    inner::O
    inner_maxiter::Int
    inner_gtol::Float64
    weights::W
    adaptive::Bool
end
function ADMM(term::Pair{<:Real, <:AbstractRegularizer}; ρ::Real = 1.0, inner::AbstractOptimizer = LBFGS(),
              inner_maxiter::Integer = 10, inner_gtol::Real = 1e-8, weights = nothing, adaptive::Bool = true)
    first(term) >= 0 || throw(ArgumentError("α must be nonnegative"))
    ρ > 0 || throw(ArgumentError("ρ must be positive"))
    inner isa Union{ADMM, ProximalGradient} && throw(ArgumentError("the inner method must be a smooth method"))
    return ADMM(Float64(first(term)), last(term), Float64(ρ), inner, Int(inner_maxiter), Float64(inner_gtol),
                weights, adaptive)
end

mutable struct _ADMMWorkspace{O}
    w::Vector{Float64}
    x::Vector{Float64}
    u::Vector{Float64}
    zold::Vector{Float64}
    xobj::O                      # F + ρ/2 ‖· - reference‖²_w
    ρ::Float64
    r::Float64                   # primal residual
    s::Float64                   # dual residual
end

_wnorm(w, x) = sqrt(sum(i -> w[i] * x[i]^2, eachindex(x)))

function _workspace(m::ADMM, obj, n)
    w = _weights(m.weights, n)
    term = TikhonovRegularizer(spdiagm(0 => w); reference = zeros(n))
    xobj = obj isa RegularizedObjective ?
           RegularizedObjective(obj.data, vcat(obj.weights, m.ρ), (obj.regularizers..., term), Float64[]) :
           RegularizedObjective(obj, [m.ρ], (term,), Float64[])
    return _ADMMWorkspace(w, zeros(n), zeros(n), zeros(n), xobj, m.ρ, NaN, NaN)
end

function _initialize!(st, ws::_ADMMWorkspace, m::ADMM, obj, box)
    F = value_and_gradient!(st.g, obj, st.σ)
    st.nevals += 1
    isfinite(F) || throw(ArgumentError("the objective is not finite at the initial guess"))
    st.value = F + _regularizer_value(m.α, m.G, st.σ)
    copyto!(ws.x, st.σ)
    fill!(ws.u, 0)
    return st
end

_optimality(st, ws::_ADMMWorkspace, ::ADMM, box) = isnan(ws.r) ? Inf : ws.r
function _gtol_met(st, ws::_ADMMWorkspace, ::ADMM, gnorm, gnorm0, gtol)
    isnan(ws.r) && return false
    w = ws.w
    tiny = sqrt(eps()) * gtol
    return ws.r <= gtol * max(_wnorm(w, ws.x), _wnorm(w, st.σ)) + tiny &&
           ws.s <= gtol * ws.ρ * _wnorm(w, ws.u) + tiny
end

function _step!(st, ws::_ADMMWorkspace, m::ADMM, obj, box)
    z, x, u, w = st.σ, ws.x, ws.u, ws.w
    term = ws.xobj.regularizers[end]
    # data step
    term.reference .= z .- u
    ws.xobj.weights[end] = ws.ρ
    res = minimize(ws.xobj, x, m.inner; lower = _bound_or_nothing(box, box.lower),
                   upper = _bound_or_nothing(box, box.upper), maxiter = m.inner_maxiter, gtol = m.inner_gtol)
    st.nevals += res.nevals
    copyto!(x, res.σ)
    Fx = res.value - ws.ρ / 2 * sum(i -> w[i] * (x[i] - term.reference[i])^2, eachindex(x))
    st.g .= res.g .- ws.ρ .* w .* (x .- term.reference)          # ∇F(x)
    # regularizer step
    copyto!(ws.zold, z)
    _scaled_prox!(z, m.α, m.G, x .+ u, ws.ρ, w, box; scale = _prox_scale(ws.r, w, x .+ u))
    # dual step and residuals
    u .+= x .- z
    ws.r = _wnorm(w, x .- z)
    ws.s = ws.ρ * _wnorm(w, z .- ws.zold)
    st.value = Fx + _regularizer_value(m.α, m.G, z)
    if m.adaptive
        if ws.r > 10ws.s
            ws.ρ *= 2
            u ./= 2
        elseif ws.s > 10ws.r
            ws.ρ /= 2
            u .*= 2
        end
    end
    return true
end
