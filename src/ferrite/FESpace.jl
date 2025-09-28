"""
    FerriteFESpace{RefElem}

Finite element space for a given reference element type `RefElem`.

# Fields
- `cellvalues::CellValues` : precomputed element-level shape functions and quadrature.
- `dh::DofHandler` : handles mapping between global and local degrees of freedom.
- `boundary_faces::Vector{Int}` : indices of boundary faces.
- `facetvalues::FacetValues` : precomputed facet-level shape functions for boundary integrals.
- `ch::ConstraintHandler` : handles Dirichlet (or other) constraints.
- `order::Int` : polynomial order of FE basis.
- `qr_order::Int` : quadrature order for integration.
- `dim::Int` : spatial dimension.
- `n::Int` : number of global DOFs.
- `m::Int` : number of boundary faces.
"""
struct FerriteFESpace{RefElem}
    cellvalues::CellValues
    dh::DofHandler
    boundary_faces::Vector{Int}
    facetvalues::FacetValues
    ch::ConstraintHandler
    order::Int
    qr_order::Int
    dim::Int
    n::Int
    m::Int
end

"""
    FerriteFESpace{RefElem}(grid, order, qr_order, boundary_faces)

Constructor for a type-stable FE space.

# Arguments
- `grid` : mesh/grid object
- `order::Int` : polynomial order
- `qr_order::Int` : quadrature order
- `boundary_faces::Vector{Int}` : indices of boundary faces for Dirichlet BCs
"""
function FerriteFESpace{RefElem}(grid, order::Int, qr_order::Int, boundary_faces::Vector{Int}) where RefElem
    dim = Ferrite.getspatialdim(grid)

    # reference element interpolation
    ip = Lagrange{RefElem, order}()
    
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
    dbc = Dirichlet(:u, boundary_faces, (x) -> 0.0)
    add!(ch, dbc)
    close!(ch)

    n = ndofs(dh)
    m = length(boundary_faces)

    return FerriteFESpace{RefElem}(cellvalues, dh, boundary_faces, facetvalues, ch, order, qr_order, dim, n, m)
end
