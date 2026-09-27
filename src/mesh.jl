"""
    EITMesh{T<:Real}

A two-dimensional triangular mesh of the measurement domain.

# Fields
- `nodes::Matrix{T}`: `2 × N` matrix of node coordinates.
- `elements::Matrix{Int}`: `3 × M` matrix of node indices, one column per triangle.
- `boundary::Vector{Int}`: indices of the nodes on the domain boundary,
  ordered counter-clockwise.

See also [`circle_mesh`](@ref).
"""
struct EITMesh{T<:Real}
    nodes::Matrix{T}
    elements::Matrix{Int}
    boundary::Vector{Int}
end

"""
    nnodes(mesh::EITMesh) -> Int

Number of nodes of `mesh`.
"""
nnodes(mesh::EITMesh) = size(mesh.nodes, 2)

"""
    nelements(mesh::EITMesh) -> Int

Number of triangles of `mesh`.
"""
nelements(mesh::EITMesh) = size(mesh.elements, 2)

"""
    circle_mesh(n::Integer; radius=1.0) -> EITMesh

Create a fan-triangulated disc with `n` boundary nodes and one centre node.

This is a mock mesh generator; a real implementation would call Gmsh.

# Examples
```jldoctest
julia> mesh = circle_mesh(16);

julia> nnodes(mesh), nelements(mesh)
(17, 16)
```
"""
function circle_mesh(n::Integer; radius::Real=1.0)
    n ≥ 3 || throw(ArgumentError("need at least 3 boundary nodes, got $n"))
    θ = range(0, 2π; length=n + 1)[1:n]
    nodes = hcat(zeros(2), radius .* vcat(cos.(θ)', sin.(θ)'))
    elements = [ones(Int, n)'; (2:n+1)'; vcat(3:n+1, 2)']
    return EITMesh(float.(nodes), elements, collect(2:n+1))
end
