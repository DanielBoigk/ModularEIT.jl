# Gridap discretization: separate finite element spaces for u and σ on one DiscreteModel.
#
# The u space is a conforming Lagrange space without Dirichlet conditions (all dofs free: the
# Neumann problem is singular, its null space is handled by the projected solvers of ModularEIT).
# The σ space is a discontinuous (`:L2`) or continuous (`:H1`) Lagrange space. Facets are
# identified by their global ids among the faces of dimension D-1 of the model; electrodes are
# vectors of such ids.

"""
    GridapDiscretization(model; order_u = 1, order_σ = 0, σ_conformity = order_σ == 0 ? :L2 : :H1,
                         boundary = nothing, degree = nothing)

Finite element spaces for the potential `u` (Lagrange of order `order_u`) and the conductivity
`σ` (Lagrange of order `order_σ`, discontinuous with `σ_conformity = :L2`, continuous with
`:H1`; default: piecewise constants) on the Gridap `DiscreteModel` `model` (e.g. a
`CartesianDiscreteModel`, `simplexify(...)` of it, or a mesh read with GridapGmsh).

- `boundary`: boundary facets (global ids of the faces of dimension D-1). Default: all facets
  that belong to exactly one cell.
- `degree`: quadrature degree. Default: exact for the weighted stiffness matrix
  `∫ σ ∇φᵢ⋅∇φⱼ` and the mass matrices on affine cells.

Facets are assumed straight (affine geometry) for facet measures and midpoints.

Fields: `model`, `V`, `U` (test and trial space of `u`), `Vσ`, `Uσ`, `Ω`, `dΩ`, `degree`,
`boundary_facets`, `boundary_dofs` (u dofs on the boundary), `order_u`, `order_σ`.
"""
struct GridapDiscretization{M, TV, TU, TS, TSU, TΩ, TdΩ} <: AbstractDiscretization
    model::M
    V::TV
    U::TU
    Vσ::TS
    Uσ::TSU
    Ω::TΩ
    dΩ::TdΩ
    degree::Int
    boundary_facets::Vector{Int}
    boundary_dofs::Vector{Int}
    order_u::Int
    order_σ::Int
end

function GridapDiscretization(model::DiscreteModel; order_u::Integer = 1, order_σ::Integer = 0,
                              σ_conformity::Symbol = order_σ == 0 ? :L2 : :H1, boundary = nothing,
                              degree = nothing)
    order_u >= 1 || throw(ArgumentError("order_u must be at least 1"))
    σ_conformity in (:L2, :H1) || throw(ArgumentError("σ_conformity must be :L2 or :H1"))
    order_σ == 0 && σ_conformity === :H1 &&
        throw(ArgumentError("piecewise constants are discontinuous: use σ_conformity = :L2"))
    V = TestFESpace(model, ReferenceFE(lagrangian, Float64, order_u); conformity = :H1)
    U = TrialFESpace(V)
    Vσ = FESpace(model, ReferenceFE(lagrangian, Float64, order_σ); conformity = σ_conformity)
    Uσ = TrialFESpace(Vσ)
    deg = degree === nothing ? max(2order_u, 2order_σ, 2(order_u - 1) + order_σ) : Int(degree)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, deg)
    D = num_cell_dims(model)
    bfacets = boundary === nothing ? findall(get_isboundary_face(get_grid_topology(model), D - 1)) :
              collect(Int, boundary)
    disc = GridapDiscretization(model, V, U, Vσ, Uσ, Ω, dΩ, deg, bfacets, Int[], Int(order_u), Int(order_σ))
    # boundary dofs: the dofs whose basis function does not vanish on the boundary
    append!(disc.boundary_dofs, _facet_free_dofs(disc, bfacets))
    return disc
end

ndofs_u(d::GridapDiscretization) = num_free_dofs(d.V)
ndofs_σ(d::GridapDiscretization) = num_free_dofs(d.Vσ)

_spatial_dim(d::GridapDiscretization) = num_point_dims(d.model)

# space of a field (:u or :σ)
_space(d::GridapDiscretization, field::Symbol) =
    field === :u ? (d.U, d.V) : field === :σ ? (d.Uσ, d.Vσ) :
    throw(ArgumentError("field must be :u or :σ, got :$field"))

# sparsity pattern of a space (all pairs of dofs that share a cell), with zero values
function _pattern(space)
    ids = get_cell_dof_ids(space)
    I, J = Int[], Int[]
    for dofs in ids, j in dofs, i in dofs
        push!(I, i)
        push!(J, j)
    end
    n = num_free_dofs(space)
    P = sparse(I, J, ones(length(I)), n, n)
    fill!(nonzeros(P), 0)
    return P
end

_u_pattern(d::GridapDiscretization) = _pattern(d.V)

# vertex coordinates of a face of dimension `dim`
function _face_coordinates(d::GridapDiscretization, dim::Integer, f::Integer)
    nodes = get_face_nodes(d.model, dim)[f]
    x = get_node_coordinates(d.model)
    return [x[n] for n in nodes]
end
