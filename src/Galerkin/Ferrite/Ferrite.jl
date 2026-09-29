# Ferrite.jl back end: discretization with separate finite element spaces for u and σ on one grid.
#
# u and σ get their own DofHandler. One DofHandler with two fields would number u and σ jointly,
# so every matrix would be (n_u + n_σ)² and the u–u block would have to be extracted. Both
# DofHandlers iterate the same grid, so a cell id gives the u dofs and the σ dofs of that cell.
# No pairing is assumed: P1/P0 is the default, but any Ferrite interpolations can be combined
# (e.g. P2/P1, or Q1/Q0 on the quadrilateral meshes of pixel images).
#
# Non-conforming meshes (hanging nodes after adaptive refinement of quadrilaterals/hexahedra)
# are supported through the conformity constraints u_full = C u: every u-space quantity
# (matrices, loads, dof lists, the conductivity tensor) is condensed to the free dofs, so
# everything downstream works with a conforming space. `ndofs_u` counts the free dofs.

using Ferrite
using SparseArrays
using LinearAlgebra

"""
    FerriteDiscretization(grid; ip_u, ip_σ, boundary, qr_order)

Finite element spaces for the potential `u` (interpolation `ip_u`, default linear Lagrange) and
the conductivity `σ` (interpolation `ip_σ`, default piecewise constant `DiscontinuousLagrange{…,0}`)
on the same Ferrite `grid`.

- `boundary`: boundary facets (a collection of `FacetIndex` or the name of a facet set). Default:
  all facets that belong to exactly one cell.
- `qr_order`: quadrature order. Default: exact for the weighted stiffness matrix
  `∫ σ ∇φᵢ⋅∇φⱼ` and the mass matrices on affine cells.

On a non-conforming grid (hanging nodes, see [`AdaptiveMesh`](@ref)) the u space is the
conforming subspace `u_full = C u`; only linear Lagrange `ip_u` and discontinuous `ip_σ` are
supported there.

Fields: `grid`, `dh_u`, `dh_σ`, `ip_u`, `ip_σ`, `cv_u`, `cv_σ` (cell values on one common
quadrature rule), `fv_u` (facet values), `boundary_facets`, `boundary_dofs` (u dofs on the
boundary), `interior_facets` (`(cell₁, facet₁, cell₂)` for every facet shared by two cells,
including the pieces of coarse–fine interfaces), `C_u` (conformity constraints
`ndofs(dh_u) × ndofs_u`, `nothing` on conforming grids).
"""
struct FerriteDiscretization{G, DHU, DHS, IPU, IPS, CVU, CVS, FVU, CT} <: AbstractDiscretization
    grid::G
    dh_u::DHU
    dh_σ::DHS
    ip_u::IPU
    ip_σ::IPS
    cv_u::CVU
    cv_σ::CVS
    fv_u::FVU
    boundary_facets::Vector{FacetIndex}
    boundary_dofs::Vector{Int}
    interior_facets::Vector{NTuple{3, Int}}
    C_u::CT
    free_u::Vector{Int}       # free dof → dof of dh_u
    full_to_free::Vector{Int} # dof of dh_u → free dof (0 for constrained dofs)
end

function FerriteDiscretization(grid::Ferrite.AbstractGrid; ip_u = nothing, ip_σ = nothing,
                               boundary = nothing, qr_order = nothing)
    shape = Ferrite.getrefshape(getcelltype(grid))
    ip_u = ip_u === nothing ? Lagrange{shape, 1}() : ip_u
    ip_σ = ip_σ === nothing ? DiscontinuousLagrange{shape, 0}() : ip_σ
    dh_u = DofHandler(grid)
    add!(dh_u, :u, ip_u)
    close!(dh_u)
    dh_σ = DofHandler(grid)
    add!(dh_σ, :σ, ip_σ)
    close!(dh_σ)

    pu, pσ = Ferrite.getorder(ip_u), Ferrite.getorder(ip_σ)
    order = qr_order === nothing ? _quadrature_order(shape, max(2pu, 2pσ, 2(pu - 1) + pσ)) : qr_order
    qr = QuadratureRule{shape}(order)
    cv_u = CellValues(qr, ip_u)
    cv_σ = CellValues(qr, ip_σ)
    fv_u = FacetValues(FacetQuadratureRule{shape}(2pu + 1), ip_u)

    C_u, free_u = _conformity(grid, dh_u)
    if C_u !== nothing
        ip_σ isa DiscontinuousLagrange ||
            throw(ArgumentError("on non-conforming grids σ must be discontinuous (e.g. piecewise constant); continuous σ spaces with hanging nodes are not implemented yet"))
    end
    full_to_free = zeros(Int, ndofs(dh_u))
    full_to_free[free_u] .= eachindex(free_u)

    bfacets, ifacets = _facet_topology(grid)
    if boundary !== nothing
        bfacets = boundary isa AbstractString ? collect(getfacetset(grid, boundary)) : collect(boundary)
    end
    bdofs = full_to_free[_facet_dofs(dh_u, ip_u, bfacets)]
    all(>(0), bdofs) || throw(ArgumentError("constrained (hanging) dofs on the boundary"))
    return FerriteDiscretization(grid, dh_u, dh_σ, ip_u, ip_σ, cv_u, cv_σ, fv_u, bfacets, bdofs, ifacets,
                                 C_u, free_u, full_to_free)
end

"""
    is_nonconforming(grid)

Whether `grid` has hanging nodes (a non-conforming grid from adaptive refinement).
"""
is_nonconforming(grid) = grid isa Ferrite.AMR.NonConformingGrid && !isempty(grid.conformity_info)

# conformity constraints u_full = C u_free on non-conforming grids
_conformity(grid, dh) = (nothing, collect(1:ndofs(dh)))
function _conformity(grid::Ferrite.AMR.NonConformingGrid, dh)
    is_nonconforming(grid) || return (nothing, collect(1:ndofs(dh)))
    ch = ConstraintHandler(dh)
    add!(ch, Ferrite.AMR.ConformityConstraint(:u))
    close!(ch)
    C, _ = Ferrite.create_constraint_matrix(ch)
    return C, copy(ch.free_dofs)
end

# u-space condensation: Cᵀ A C for matrices, Cᵀ f for load vectors (identity when conforming)
_condense(d::FerriteDiscretization, A::AbstractMatrix) = d.C_u === nothing ? A : sparse(d.C_u' * A * d.C_u)
_condense(d::FerriteDiscretization, f::AbstractVector) = d.C_u === nothing ? f : d.C_u' * f
# nodal values of dh_u → free dofs (the constrained values are implied by C)
_restrict(d::FerriteDiscretization, v::AbstractVector) = d.C_u === nothing ? v : v[d.free_u]
# full u vector (for element loops) from free coefficients
_lift(d::FerriteDiscretization, v::AbstractVecOrMat) = d.C_u === nothing ? v : d.C_u * v
# sparsity pattern of the u space
function _u_pattern(d::FerriteDiscretization)
    P = allocate_matrix(d.dh_u)
    d.C_u === nothing && return P
    fill!(nonzeros(P), 1)
    Pc = sparse(abs.(d.C_u') * P * abs.(d.C_u))
    fill!(nonzeros(Pc), 0)
    return Pc
end

# Ferrite's `order` is the polynomial degree on simplices and the number of Gauss points per
# direction on hypercubes (exact for per-direction degree 2n - 1).
_quadrature_order(::Type{<:Ferrite.RefSimplex}, degree) = max(degree, 1)
_quadrature_order(::Type{<:Ferrite.RefHypercube}, degree) = max(cld(degree + 1, 2), 1)

"""
    ndofs_u(disc)

Number of degrees of freedom of the potential.
"""
ndofs_u(d::FerriteDiscretization) = length(d.free_u)

"""
    ndofs_σ(disc)

Number of degrees of freedom of the conductivity.
"""
ndofs_σ(d::FerriteDiscretization) = ndofs(d.dh_σ)

# Boundary facets (facets of exactly one cell) and interior facets (cell₁, local facet₁, cell₂).
# On non-conforming 2D grids a coarse facet is split by a hanging node into two fine facets; each
# fine facet is paired with the coarse cell (the hanging node's masters are the coarse facet's
# end points).
function _facet_topology(grid)
    seen = Dict{Vector{Int}, Tuple{Int, Int}}()
    interior = NTuple{3, Int}[]
    for (ci, cell) in enumerate(getcells(grid)), (lf, nodes) in enumerate(Ferrite.facets(cell))
        key = sort!(collect(nodes))
        other = pop!(seen, key, nothing)
        if other === nothing
            seen[key] = (ci, lf)
        else
            push!(interior, (other[1], other[2], ci))
        end
    end
    if is_nonconforming(grid)
        info = grid.conformity_info
        paired = Set{Vector{Int}}()
        for (key, (ci, lf)) in seen
            for h in key
                haskey(info, h) || continue
                coarse = sort(collect(info[h]))
                haskey(seen, coarse) || continue
                push!(interior, (ci, lf, seen[coarse][1]))
                push!(paired, key, coarse)
            end
        end
        foreach(k -> delete!(seen, k), paired)
    end
    boundary = sort!([FacetIndex(c, f) for (c, f) in values(seen)]; by = fi -> fi.idx)
    return boundary, interior
end

# u dofs on a set of facets (vertices, edges and facet interiors), sorted
function _facet_dofs(dh::DofHandler, ip, facets)
    local_dofs = Ferrite.dirichlet_facetdof_indices(ip)
    dofs = Set{Int}()
    buf = zeros(Int, ndofs_per_cell(dh))
    for fi in facets
        c, f = fi.idx
        celldofs!(buf, dh, c)
        for i in local_dofs[f]
            push!(dofs, buf[i])
        end
    end
    return sort!(collect(dofs))
end

# measure (length/area) of one facet
function _facet_measure(disc::FerriteDiscretization, fi::FacetIndex)
    c, f = fi.idx
    reinit!(disc.fv_u, getcoordinates(disc.grid, c), f)
    return sum(q -> getdetJdV(disc.fv_u, q), 1:getnquadpoints(disc.fv_u))
end

# midpoint of a facet (mean of its vertex coordinates)
function _facet_midpoint(grid, fi::FacetIndex)
    c, f = fi.idx
    nodes = Ferrite.facets(getcells(grid, c))[f]
    return sum(n -> get_node_coordinate(grid, n), nodes) / length(nodes)
end

include("Assemblers/MatrixAssemblers.jl")
include("Assemblers/TensorAssembler.jl")
include("Assemblers/CoeffAssembler.jl")
include("FESpace.jl")
include("StructuredGrid.jl")
include("PolarGrid.jl")
include("Regularizers.jl")
include("Boundary.jl")
include("PatternSVD.jl")
include("Images.jl")
include("Bisection.jl")
include("AdaptiveMeshing.jl")
include("ResidualEstimator.jl")
