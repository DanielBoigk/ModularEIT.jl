"""
    Electrode{T<:Real}

An electrode of the Complete Electrode Model.

# Fields
- `nodes::Vector{Int}`: boundary nodes covered by the electrode.
- `impedance::T`: contact impedance ``z_\\ell > 0``.
"""
struct Electrode{T<:Real}
    nodes::Vector{Int}
    impedance::T
end

"""
    ring_electrodes(mesh::EITMesh, L::Integer; impedance=1e-2) -> Vector{Electrode}

Place `L` equally spaced point electrodes on the boundary of `mesh`.

# Examples
```jldoctest
julia> els = ring_electrodes(circle_mesh(32), 8);

julia> length(els)
8
```
"""
function ring_electrodes(mesh::EITMesh, L::Integer; impedance::Real=1e-2)
    nb = length(mesh.boundary)
    L ≤ nb || throw(ArgumentError("cannot place $L electrodes on $nb boundary nodes"))
    idx = round.(Int, range(1, nb + 1; length=L + 1)[1:L])
    return [Electrode([mesh.boundary[i]], float(impedance)) for i in idx]
end
