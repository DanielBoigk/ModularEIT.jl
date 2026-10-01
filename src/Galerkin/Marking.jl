# Marking strategies of adaptive mesh refinement (independent of the mesh and the back end).

"""
    dorfler_marking(η, θ)

Dörfler (bulk) marking: the smallest set of cells, taken in decreasing order of `η`, whose
indicators sum to at least `θ` times the total.
"""
function dorfler_marking(η::AbstractVector, θ::Real)
    0 < θ <= 1 || throw(ArgumentError("θ must be in (0, 1]"))
    total = sum(η)
    marked = Int[]
    total > 0 || return marked
    acc = zero(total)
    for c in sortperm(η; rev = true)
        push!(marked, c)
        acc += η[c]
        acc >= θ * total && break
    end
    return marked
end
