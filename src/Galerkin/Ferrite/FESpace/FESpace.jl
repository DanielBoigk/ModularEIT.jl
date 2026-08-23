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
function FerriteFESpace{RefElem}(grid, order::Int, order_σ::Int, qr_order::Int, ∂Ω) where {RefElem}
    dim = Ferrite.getspatialdim(grid)

    # reference element interpolation
    # Ferrite's `Lagrange` only supports order ≥ 1 (continuous nodal basis);
    # order 0 (piecewise constant) is `DiscontinuousLagrange` instead. `u`
    # must stay continuous (it needs gradients and boundary traces), but σ
    # is commonly order 0 to match per-cell/per-pixel conductivity data.
    ip = Lagrange{RefElem,order}()
    σp = order_σ == 0 ? DiscontinuousLagrange{RefElem,0}() : Lagrange{RefElem,order_σ}()
    # quadrature
    qr = QuadratureRule{RefElem}(qr_order)
    qr_face = FacetQuadratureRule{RefElem}(qr_order)

    # cell and facet values
    cellvalues = CellValues(qr, ip)
    facetvalues = FacetValues(qr_face, ip)

    cellvalues_σ = CellValues(qr, σp)

    # degrees of freedom
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)

    # σ (conductivity) lives on its own DofHandler, since it may use a
    # different polynomial order than u and must not perturb the sizing
    # of the potential-field operators (n, M, K, up/down, ...).
    dh_σ = DofHandler(grid)
    add!(dh_σ, :σ, σp)
    close!(dh_σ)
    n_σ = ndofs(dh_σ)

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

    m, _, down, up, up!, _, b_dofs = produce_nonzero_positions(facetvalues, dh, ∂Ω)
    BDO = BoundaryOperators(nothing,nothing,nothing,nothing,nothing,nothing)
    return FerriteFESpace{RefElem}(cellvalues, dh, ∂Ω, facetvalues, ch, order, qr_order, dim, n, num_facet, m, M, M_fac, K, K_fac, total_volume, b_dofs, down, up, up!, BDO, cellvalues_σ, dh_σ, n_σ)
end

include("Norms.jl")
include("Operators.jl")

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
