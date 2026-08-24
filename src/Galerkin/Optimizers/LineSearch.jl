# Stated purpose:
# after Gauss-Newton search for
# https://www.sciencedirect.com/science/article/pii/S0898122117302833 (page 4)
#

using Optim, LineSearches

"""
    determine_box(σ, δ; lb=1e-6, ub=nothing, ϵ=1e-12)

Largest step-size interval `[τ_min, τ_max]` such that *every* component of
`σ .+ τ .* δ` stays within `[lb, ub]` (or just `≥ lb` when `ub === nothing`),
found by intersecting each component's own feasible τ-interval. Components
with `|δ_i| < ϵ` barely move regardless of τ and are treated as unconstrained.

This is an exact box (unlike bounding only `mean(σ .+ τ .* δ)`, which neither
guarantees nor tightly bounds what any individual component does).
"""
function determine_box(σ::AbstractVector, δ::AbstractVector; lb::Number=1e-6, ub::Union{Number,Nothing}=nothing, ϵ::Number=1e-12)
    ub_eff = ub === nothing ? Inf : ub
    τ_min, τ_max = -Inf, Inf
    @inbounds for i in eachindex(σ, δ)
        δi = δ[i]
        abs(δi) < ϵ && continue
        lo = (lb - σ[i]) / δi
        hi = (ub_eff - σ[i]) / δi
        a, b = δi > 0 ? (lo, hi) : (hi, lo)
        τ_min = max(τ_min, a)
        τ_max = min(τ_max, b)
    end
    return τ_min, τ_max
end

