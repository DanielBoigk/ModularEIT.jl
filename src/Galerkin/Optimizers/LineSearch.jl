# Stated purpose:
# after Gauss-Newton search for
# https://www.sciencedirect.com/science/article/pii/S0898122117302833 (page 4)
#

using Optim, LineSearches

# Idea: later for the line search we
function determine_box(σ::AbstractVector, δ::AbstractVector, max::Number=1)
    σ_mean = Statistics.mean(σ)
    δ_mean = Statistics.mean(δ)
    τ_max = (max - σ_mean) / δ_mean
    τ_min = (-σ_mean) / δ_mean
    if τ_max < τ_min
        σ_mean = τ_max
        τ_max = τ_min
        τ_min = σ_mean
    end
    τ_min, τ_max
end

