# Differentiation rules for ChainRulesCore-based AD (Zygote, Diffractor; Enzyme via
# `Enzyme.@import_rrule`): ModularEIT functions can be used inside differentiated programs, e.g.
# a conductivity produced by a neural network, or training through the forward model. The
# derivatives come from the adjoint-state and linearized solves of ModularEIT, not from
# differentiating through the finite element and linear solver code.
module ModularEITChainRulesCoreExt

using ModularEIT
using ChainRulesCore
import ModularEIT: AbstractObjective, objective_value, value_and_gradient!, residual, residual!,
    n_residual, residual_and_jacobian!, jacobian_operator, _require_coefficient_gradient

# J(σ), pullback J̄ ↦ J̄ ∇J(σ) (the adjoint-state coefficient gradient)
function ChainRulesCore.rrule(::typeof(objective_value), obj::AbstractObjective, σ::AbstractVector)
    _require_coefficient_gradient(obj)
    g = zeros(length(σ))
    J = value_and_gradient!(g, obj, σ)
    function objective_value_pullback(J̄)
        J̄ = unthunk(J̄)
        return NoTangent(), NoTangent(), J̄ isa AbstractZero ? ZeroTangent() : J̄ .* g
    end
    return J, objective_value_pullback
end

# a reconstruction problem is differentiated as its objective
ChainRulesCore.rrule(::typeof(objective_value), prob::ModularEIT.EITProblem, θ::AbstractVector) =
    ChainRulesCore.rrule(objective_value, prob.objective, θ)

# The Jacobian of the residual at σ: matrix-free where available, dense otherwise.
function _jacobian(obj, σ)
    try
        return jacobian_operator(obj, σ)
    catch e
        e isa Union{MethodError, ArgumentError} || rethrow()
    end
    J = zeros(n_residual(obj), length(σ))
    residual_and_jacobian!(zeros(n_residual(obj)), J, obj, σ)
    return J
end

# r(σ), pullback r̄ ↦ Jᵀ r̄ (one adjoint solve per pattern)
function ChainRulesCore.rrule(::typeof(residual), obj::AbstractObjective, σ::AbstractVector)
    r = residual(obj, σ)
    Jσ = _jacobian(obj, σ)
    function residual_pullback(r̄)
        r̄ = unthunk(r̄)
        return NoTangent(), NoTangent(), r̄ isa AbstractZero ? ZeroTangent() : Jσ' * collect(Float64, r̄)
    end
    return r, residual_pullback
end

# r(σ), pushforward δσ ↦ J δσ (one linearized solve per pattern)
function ChainRulesCore.frule((_, _, δσ), ::typeof(residual), obj::AbstractObjective, σ::AbstractVector)
    r = residual(obj, σ)
    δσ = unthunk(δσ)
    δσ isa AbstractZero && return r, ZeroTangent()
    return r, _jacobian(obj, σ) * collect(Float64, δσ)
end

end
