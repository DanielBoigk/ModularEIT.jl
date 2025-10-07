using Ferrite

export produce_nonzero_positions
function produce_nonzero_positions(v, atol=1e-8, rtol=1e-5)
    approx_zero(x; atol=atol, rtol=rtol) = isapprox(x, 0; atol=atol, rtol=rtol)
    non_zero_count = count(x -> !approx_zero(x), v)
    non_zero_positions = zeros(Int, non_zero_count)
    non_zero_indices = findall(x -> !approx_zero(x), v)
    down = (x) -> x[non_zero_indices]
    up = (x) -> begin
        v = zeros(eltype(x), length(v))
        v[non_zero_indices] = x
        return v
    end
    return non_zero_count, non_zero_positions, down, up
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
    nzc, nzpos, down, up = produce_nonzero_positions(f)
    return nzc, nzpos, down, up, f
end
function produce_nonzero_positions(fe::FerriteFESpace)
    facetvalues = fe.facetvalues
    dh = fe.dh
    ∂Ω = fe.∂Ω
    produce_nonzero_positions(facetvalues, dh, ∂Ω)
end
