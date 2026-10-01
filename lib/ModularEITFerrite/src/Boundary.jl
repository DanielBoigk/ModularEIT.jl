# Electrode primitives of the Ferrite back end (see Galerkin/Electrodes.jl in ModularEIT, which
# builds electrode placement, grounding and the forward models of the electrode models from
# them). Electrodes are vectors of boundary `FacetIndex`.

_boundary_facets(d::FerriteDiscretization) = d.boundary_facets
_boundary_dofs(d::FerriteDiscretization) = d.boundary_dofs
_boundary_mass(d::FerriteDiscretization, facets) = _condense(d, assemble_boundary_mass(d.dh_u, d.fv_u, facets))
_boundary_load(d::FerriteDiscretization, facets) =
    _condense(d, assemble_boundary_load!(zeros(ndofs(d.dh_u)), d.dh_u, d.fv_u, facets))
_facet_free_dofs(d::FerriteDiscretization, facets) = d.full_to_free[_facet_dofs(d.dh_u, d.ip_u, facets)]
_facet_midpoint(d::FerriteDiscretization, fi::FacetIndex) = _facet_midpoint(d.grid, fi)
function _facet_vertices(d::FerriteDiscretization, fi::FacetIndex)
    c, f = fi.idx
    return Tuple(get_node_coordinate(d.grid, n) for n in Ferrite.facets(getcells(d.grid, c))[f])
end
_spatial_dim(d::FerriteDiscretization) = Ferrite.getspatialdim(d.grid)
