using Ferrite
export FerriteFESpace

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
struct FerriteFESpace{RefElem} <: AbstractHilbertSpace
    cellvalues::CellValues
    dh::DofHandler
    boundary_faces
    facetvalues::FacetValues
    ch::ConstraintHandler
    order::Int
    qr_order::Int
    dim::Int
    n::Int
    m::Int
    M::AbstractMatrix
    M_fac
    K::AbstractMatrix
    K_fac
    total_volume::Float64
end

"""
    FerriteFESpace{RefElem}(grid, order, qr_order, boundary_faces)

Constructor for a type-stable FE space.

# Arguments
- `grid` : mesh/grid object
- `order::Int` : polynomial order
- `qr_order::Int` : quadrature order
- `boundary_faces` : indices of boundary faces for Dirichlet BCs
"""
function FerriteFESpace{RefElem}(grid, order::Int, qr_order::Int, boundary_faces) where {RefElem}
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
    dbc = Dirichlet(:u, boundary_faces, (x) -> 0.0)
    add!(ch, dbc)
    close!(ch)

    n = ndofs(dh)
    m = length(boundary_faces)
    M, M_fac = assemble_M(dh, cellvalues)
    K, K_fac = assemble_K(dh, cellvalues)
    total_volume = calc_total_volume(dh, cellvalues)
    return FerriteFESpace{RefElem}(cellvalues, dh, boundary_faces, facetvalues, ch, order, qr_order, dim, n, m, M, M_fac, K, K_fac, total_volume)
end

function dotH1(fe::FerriteFESpace, a::AbstractVector, b::AbstractVector)
    return a' * fe.K * b
end

function dotL2(fe::FerriteFESpace, a::AbstractVector, b::AbstractVector)
    return a' * fe.M * b
end

function dotL2(fe::FerriteFESpace, a::AbstractVector)
    return a' * fe.M * a
end

function dotH1(fe::FerriteFESpace, a::AbstractVector)
    return a' * fe.K * a
end

function normH1sq(fe::FerriteFESpace, a::AbstractVector)
    return dotH1(fe, a, a)
end
function normL2sq(fe::FerriteFESpace, a::AbstractVector)
    return dotL2(fe, a, a)
end
function normL2(fe::FerriteFESpace, a::AbstractVector)
    return sqrt(normL2sq(fe, a))
end
function normH1(fe::FerriteFESpace, a::AbstractVector)
    return sqrt(normH1(fe, a))
end
function metricL2sq(fe::FerriteFESpace, a::AbstractVector, b::AbstractVector)
    return normL2sq(fe, a - b)
end
function metricL2(fe::FerriteFESpace, a::AbstractVector, b::AbstractVector)
    return normL2(fe, a - b)
end
function metricH1(fe::FerriteFESpace, a::AbstractVector, b::AbstractVector)
    return normH1(fe, a - b)
end
function metricH1sq(fe::FerriteFESpace, a::AbstractVector, b::AbstractVector)
    return normH1sq(fe, a - b)
end

function normL1(a::AbstractVector, cellvalues::CellValues, dh::DofHandler)
    n_basefuncs = getnbasefunctions(cellvalues)
    qpoints = getnquadpoints(cellvalues)
    total_residual = 0.0
    total_volume = 0.0
    for cell in CellIterator(dh)
        dofs = celldofs(cell)
        reinit!(cellvalues, cell)
        ue = a[dofs]
        for q in 1:qpoints
            dΩ = getdetJdV(cellvalues, q)
            total_volume += dΩ

            uh_q = 0.0
            for i in 1:n_basefuncs
                ϕᵢ = shape_function(cellvalues, q, i)
                uh_q += ue[i] * ϕᵢ
            end

            total_residual += abs(uh_q) * dΩ
        end
    end
    return total_residual, total_volume
end
function normL1(fe::FerriteFESpace, a::AbstractVector)
    normL1(a, fe.cellvalues, fe.dh)
end

function normL1grad(a::AbstractVector, cellvalues::CellValues, dh::DofHandler, ndims::Int64)
    n_basefuncs = getnbasefunctions(cellvalues)
    qpoints = getnquadpoints(cellvalues)
    total_residual = 0.0
    for cell in CellIterator(dh)
        dofs = celldofs(cell)
        reinit!(cellvalues, cell)
        ue = a[dofs]
        for q in 1:qpoints
            dΩ = getdetJdV(cellvalues, q)
            ∇uh_q = zeros(eltype(a), ndims)
            for i in 1:n_basefuncs
                ∇ϕᵢ = shape_gradient(cellvalues, q, i)
                ∇uh_q .+= ue[i] * ∇ϕᵢ
            end

            total_residual += norm(∇uh_q) * dΩ
        end
    end

    return total_residual
end
function normL2Grad(fe::FerriteFESpace, a::AbstractVector)
    normL1grad(a, fe.cellvalues, fe.dh, fe.dim)
end

function calc_total_volume(dh::DofHandler, cellvalues::CellValues)
    total_volume = 0.0
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        for q in 1:qpoints
            total_volume += getdetJdV(cellvalues, q)
        end
    end
    return total_volume
end
