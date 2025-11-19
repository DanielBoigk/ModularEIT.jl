using Ferrite
using LinearAlgebra

export FerriteFESpace






"""
    FerriteFESpace{RefElem}(grid, order, qr_order, ∂Ω)

Constructs a type-stable finite element space.

# Arguments
- `grid` : mesh/grid object
- `order::Int` : polynomial order of the FE basis
- `qr_order::Int` : quadrature order
- `∂Ω` : indices of boundary faces for Dirichlet conditions

Initializes cell/facet values, DOFs, constraints, mass/stiffness matrices,
and projection operators.
"""
function FerriteFESpace{RefElem}(grid, order::Int, qr_order::Int, ∂Ω) where {RefElem}
    dim = Ferrite.getspatialdim(grid)

    # reference element interpolation
    ip = Lagrange{RefElem,order}()

    # quadrature
    qr = QuadratureRule{RefElem}(qr_order)
    qr_face = FacetQuadratureRule{RefElem}(qr_order)

    # cell and facet values
    cellvalues = CellValues(qr, ip)
    facetvalues = FacetValues(qr_face, ip)

    # degrees of freedom
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)

    # constraints
    ch = ConstraintHandler(dh)
    dbc = Dirichlet(:u, ∂Ω, (x) -> 0.0)
    add!(ch, dbc)
    close!(ch)

    n = ndofs(dh)
    num_facet = length(∂Ω)
    M, M_fac = assemble_M(dh, cellvalues)
    K, K_fac = assemble_K(dh, cellvalues)
    total_volume = calc_total_volume(dh, cellvalues)

    m, _, down, up, up!, _ = produce_nonzero_positions(facetvalues, dh, ∂Ω)
    return FerriteFESpace{RefElem}(cellvalues, dh, ∂Ω, facetvalues, ch, order, qr_order, dim, n, num_facet, m, M, M_fac, K, K_fac, total_volume, down, up, up!)
end

include("Norms.jl")

function calc_total_volume(dh::DofHandler, cellvalues::CellValues)
    total_volume = 0.0
    qpoints = getnquadpoints(cellvalues)
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        for q in 1:qpoints
            total_volume += getdetJdV(cellvalues, q)
        end
    end
    return total_volume
end
