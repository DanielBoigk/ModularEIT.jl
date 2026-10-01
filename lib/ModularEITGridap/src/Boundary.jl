# Electrode primitives of the Gridap back end (see Galerkin/Electrodes.jl in ModularEIT).
# Electrodes are vectors of global facet ids.

_boundary_facets(d::GridapDiscretization) = d.boundary_facets
_boundary_dofs(d::GridapDiscretization) = d.boundary_dofs

_boundary_measure(d::GridapDiscretization, facets) =
    Measure(BoundaryTriangulation(d.model, collect(Int, facets)), 2d.order_u)

function _boundary_mass(d::GridapDiscretization, facets)
    dΓ = _boundary_measure(d, facets)
    return _csc(assemble_matrix((u, v) -> ∫(u * v)dΓ, d.U, d.V))
end

function _boundary_load(d::GridapDiscretization, facets)
    dΓ = _boundary_measure(d, facets)
    return Vector{Float64}(assemble_vector(v -> ∫(v)dΓ, d.V))
end

# the dofs whose basis function does not vanish on the facets: a clearly positive diagonal of the
# boundary mass matrix (basis functions of other dofs have traces at round-off level there)
function _facet_free_dofs(d::GridapDiscretization, facets)
    m = diag(_boundary_mass(d, facets))
    return findall(>(sqrt(eps()) * maximum(m; init = 0.0)), m)
end

function _facet_measure(d::GridapDiscretization, f::Integer)
    x = _face_coordinates(d, num_cell_dims(d.model) - 1, f)
    length(x) == 2 && return norm(x[2] - x[1])                       # segment
    length(x) == 3 && return norm(_cross(x[2] - x[1], x[3] - x[1])) / 2  # triangle
    length(x) == 4 && return norm(_cross(x[2] - x[1], x[3] - x[1]))     # parallelogram (Gridap order)
    throw(ArgumentError("facets with $(length(x)) vertices are not supported"))
end

_cross(a, b) = (a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])

function _facet_midpoint(d::GridapDiscretization, f::Integer)
    x = _face_coordinates(d, num_cell_dims(d.model) - 1, f)
    return sum(x) / length(x)
end

_facet_vertices(d::GridapDiscretization, f::Integer) = Tuple(_face_coordinates(d, num_cell_dims(d.model) - 1, f))
