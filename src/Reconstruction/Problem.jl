# A reconstruction problem as one object: discretization, forward model, data term, full
# objective (data + regularization, optionally in parameters θ with σ = P θ), the current iterate
# and the history of the runs. `reconstruct!` runs a method from the current iterate (warm start)
# and stores the result; the problem also acts as an objective in θ (value, gradient, and with
# ChainRulesCore a differentiation rule).

"""
    EITProblem(disc, model, currents, voltages; noise = nothing, τ = 1.1, regularization = (),
               parametrization = nothing, initial = 1.0, lower = 0.05, upper = nothing,
               solver = DirectSolver(), misfit = SquaredEuclidean(), grounding = :integral)

A complete EIT reconstruction problem: the discretization `disc`, the forward model of the
electrode model `model` (or a [`ForwardModel`](@ref) passed as `model`), the measured `voltages`
for the injected `currents`, and the current iterate.

- `noise`: noise model of the data. Sets the discrepancy target `τ² E[misfit of exact data]`
  (field `target`, `NaN` without a noise model), at which [`reconstruct!`](@ref) stops.
- `regularization`: terms `α => R` added to the data misfit ([`RegularizedObjective`](@ref)).
- `parametrization`: unknowns `θ` with `σ = P θ` (e.g. [`SubspaceParametrization`](@ref) or the
  pixels of a back end); bounds, `initial` and regularizers then refer to `θ`.
- `initial`: initial iterate (a number for a constant, or a vector).
- `lower`, `upper`: bounds of the iterate.
- `solver`, `misfit`, `grounding`: passed to the [`AdjointStateObjective`](@ref) and the forward
  model.

Fields: `disc`, `forward`, `data` (data term in the unknowns), `objective` (full objective),
`parametrization`, `θ` (current iterate), `target`, `lower`, `upper`, `history` (the
[`OptimizationState`](@ref) of every run).

See [`reconstruct!`](@ref), [`solution`](@ref) and [`data_misfit`](@ref). `objective_value(prob, θ)`
and `value_and_gradient!(g, prob, θ)` evaluate the full objective, so a problem can be passed
wherever an objective is used; `objective_value(prob)` evaluates it at the current iterate.
"""
mutable struct EITProblem{D, F, O, J, P} <: AbstractEITProblem
    disc::D
    forward::F
    data::O
    objective::J
    parametrization::P
    θ::Vector{Float64}
    target::Float64
    lower::Any
    upper::Any
    history::Vector{OptimizationState}
end

function EITProblem(disc::AbstractDiscretization, model, currents::AbstractVecOrMat, voltages::AbstractVecOrMat;
                    noise = nothing, τ::Real = 1.1, regularization = (), parametrization = nothing,
                    initial = 1.0, lower = 0.05, upper = nothing, solver::AbstractLinearSolver = DirectSolver(),
                    misfit::AbstractMisfit = SquaredEuclidean(), grounding::Symbol = :integral)
    fm = model isa ForwardModel ? model : ForwardModel(disc, model; grounding)
    adjoint = AdjointStateObjective(fm, currents, voltages; solver, misfit)
    target = noise === nothing ? NaN : discrepancy_target(adjoint, noise; τ)
    data = parametrization === nothing ? adjoint : ParametrizedObjective(adjoint, parametrization)
    terms = Tuple(regularization)
    objective = isempty(terms) ? data : RegularizedObjective(data, terms...)
    n = parametrization === nothing ? ndofs_σ(disc) : parameter_count(parametrization)
    θ = initial isa Real ? fill(Float64(initial), n) : Vector{Float64}(initial)
    length(θ) == n || throw(DimensionMismatch("the initial iterate needs $n entries, got $(length(θ))"))
    return EITProblem(disc, fm, data, objective, parametrization, θ, Float64(target), lower, upper,
                      OptimizationState[])
end

"""
    solution(prob::EITProblem)

The conductivity coefficients of the current iterate (`P θ` with a parametrization, otherwise
`θ`).
"""
solution(prob::EITProblem) = prob.parametrization === nothing ? copy(prob.θ) : conductivity(prob.parametrization, prob.θ)

"""
    data_misfit(prob::EITProblem, θ = prob.θ)

The data misfit (without regularization) at `θ`, the quantity that the discrepancy principle
compares with `prob.target`.
"""
data_misfit(prob::EITProblem, θ::AbstractVector = prob.θ) = objective_value(prob.data, θ)

objective_value(prob::EITProblem, θ::AbstractVector) = objective_value(prob.objective, θ)
objective_value(prob::EITProblem) = objective_value(prob, prob.θ)
value_and_gradient!(g::AbstractVector, prob::EITProblem, θ::AbstractVector) = value_and_gradient!(g, prob.objective, θ)

# proximal methods carry the non-smooth term themselves and minimize the data term
_is_proximal(method) = method isa Union{ProximalGradient, ADMM}

"""
    reconstruct!(prob::EITProblem, method = GaussNewton(); maxiter = 50, callback = nothing, kwargs...)

Minimize the objective of `prob` with `method` (any [`AbstractOptimizer`](@ref)), starting from
the current iterate (a warm start, so repeated calls continue where the last one stopped), store
the result as the new iterate, append the [`OptimizationState`](@ref) to `prob.history` and
return it. `kwargs` are passed to [`minimize`](@ref).

With a noise model the run stops by the discrepancy principle, as soon as the data misfit is
below `prob.target` (status `:ftarget`); with regularization terms this is checked on the data
misfit alone, at the cost of one forward solve per iteration. Proximal methods
([`ProximalGradient`](@ref), [`ADMM`](@ref)) carry their non-smooth term themselves and minimize
the data term `prob.data`.
"""
function reconstruct!(prob::EITProblem, method::AbstractOptimizer = GaussNewton(); maxiter::Integer = 50,
                      callback = nothing, kwargs...)
    obj = _is_proximal(method) ? prob.data : prob.objective
    stop_on_data = !isnan(prob.target) && obj !== prob.data
    ftarget = !isnan(prob.target) && !stop_on_data ? prob.target : -Inf
    reached = Ref(false)
    cb = function (st)
        callback !== nothing && callback(st) && return true
        stop_on_data || return false
        reached[] = data_misfit(prob, st.σ) <= prob.target
        return reached[]
    end
    # already explained to the noise level: nothing to do
    if !isnan(prob.target) && data_misfit(prob) <= prob.target
        st = OptimizationState(copy(prob.θ), zeros(length(prob.θ)), objective_value(obj, prob.θ), 0, 1,
                               :ftarget, true, [])
        push!(prob.history, st)
        return st
    end
    st = minimize(obj, prob.θ, method; lower = prob.lower, upper = prob.upper, maxiter, ftarget, callback = cb,
                  kwargs...)
    if reached[]
        st.status = :ftarget
        st.converged = true
    end
    prob.θ = copy(st.σ)
    push!(prob.history, st)
    return st
end

function Base.show(io::IO, prob::EITProblem)
    n_obs, s = size(prob.data isa ParametrizedObjective ? prob.data.obj.data : prob.data.data)
    ratio = isnan(prob.target) ? "" : ", misfit/target " * string(round(data_misfit(prob) / prob.target; sigdigits = 3))
    print(io, "EITProblem: ", length(prob.θ), " unknowns, ", n_obs, " × ", s, " data, ",
          length(prob.history), " run", length(prob.history) == 1 ? "" : "s", ratio)
end
