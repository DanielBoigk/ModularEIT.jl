using Ferrite

export produce_nonzero_positions
#=
function produce_nonzero_positions(facetvalues::FacetValues, dh::DofHandler, ∂Ω)
    boundary_dofs = Int[]
    for facet in FacetIterator(dh, ∂Ω)
        append!(boundary_dofs, celldofs(facet))
    end
    boundary_dofs = unique(boundary_dofs)
    f = zeros(ndofs(dh))
    f[boundary_dofs] .= 1.0
    nzc = length(boundary_dofs)
    down = (x) -> x[boundary_dofs]
    up = (x) -> begin
        v = zeros(eltype(x), ndofs(dh))
        v[boundary_dofs] = x
        v
    end
    return nzc, boundary_dofs, down, up, f
end
=#

function produce_nonzero_positions(v, atol=1e-8, rtol=1e-5)
    approx_zero(x; atol=atol, rtol=rtol) = isapprox(x, 0; atol=atol, rtol=rtol)
    non_zero_count = count(x -> !approx_zero(x), v)
    non_zero_positions = zeros(Int, non_zero_count)
    non_zero_indices = findall(x -> !approx_zero(x), v)
    n = length(v)

    # `down`/`up`/`up!` each get a matrix method alongside the vector one, so
    # they batch over modes (columns) in a single call: e.g. `up(G)` for
    # `G::AbstractMatrix` of size (non_zero_count, num_modes) lifts every
    # column at once instead of looping `up.(eachcol(G))`.
    down(x::AbstractVector) = x[non_zero_indices]
    down(X::AbstractMatrix) = X[non_zero_indices, :]

    up(x::AbstractVector) = begin
        out = zeros(eltype(x), n)
        out[non_zero_indices] = x
        return out
    end
    up(X::AbstractMatrix) = begin
        out = zeros(eltype(X), n, size(X, 2))
        out[non_zero_indices, :] = X
        return out
    end

    up!(out::AbstractVector, x::AbstractVector) = begin
        @assert length(out) == n
        @assert length(x) == non_zero_count
        fill!(out, zero(eltype(out)))          # reset (optional)
        @inbounds for (i, idx) in enumerate(non_zero_indices)
            out[idx] = x[i]
        end
        return out
    end
    up!(out::AbstractMatrix, X::AbstractMatrix) = begin
        @assert size(out, 1) == n
        @assert size(X, 1) == non_zero_count
        @assert size(out, 2) == size(X, 2)
        fill!(out, zero(eltype(out)))
        @views out[non_zero_indices, :] .= X
        return out
    end

    return non_zero_count, non_zero_positions, down, up, up!, non_zero_indices
end

function produce_nonzero_positions(facetvalues::FacetValues, dh::DofHandler, ∂Ω)
    f = zeros(ndofs(dh))
    for facet in FacetIterator(dh, ∂Ω)
        fe = zeros(ndofs_per_cell(dh))
        reinit!(facetvalues, facet)
        for q_point in 1:getnquadpoints(facetvalues)
            dΓ = getdetJdV(facetvalues, q_point)
            for i in 1:getnbasefunctions(facetvalues)
                δu = shape_value(facetvalues, q_point, i)
                fe[i] += δu * dΓ
            end
        end
        assemble!(f, celldofs(facet), fe)
    end
    nzc, nzpos, down, up, up!, nzindzs = produce_nonzero_positions(f)
    return nzc, nzpos, down, up, up!, f, nzindzs
end
function produce_nonzero_positions(fe::FerriteFESpace)
    facetvalues = fe.facetvalues
    dh = fe.dh
    ∂Ω = fe.∂Ω
    produce_nonzero_positions(facetvalues, dh, ∂Ω)
end
