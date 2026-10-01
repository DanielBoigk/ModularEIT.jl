"""
    ModularEITGridap

[Gridap.jl](https://github.com/gridap/Gridap.jl) back end of ModularEIT: finite element
discretizations of a Gridap `DiscreteModel` with separate spaces for the potential (Lagrange,
any order) and the conductivity (piecewise constants by default, or Lagrange). It implements the
back end contract of ModularEIT, so forward models of all electrode models, objectives,
regularizers, optimizers and linear solvers work unchanged on Gridap meshes.
"""
module ModularEITGridap

using LinearAlgebra
using SparseArrays
using Gridap
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.CellData
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Fields
import Gridap.TensorValues
using ModularEIT

# back end contract of ModularEIT
import ModularEIT: AbstractDiscretization, ConductivityTensor, FEMatrices, ndofs_u, ndofs_σ,
    interpolate_function, l2_project, fe_inner, fe_norm, total_variation, total_variation!,
    lumped_mass, assemble_weighted_stiffness
import ModularEIT: _boundary_facets, _boundary_dofs, _boundary_mass, _boundary_load, _facet_free_dofs,
    _facet_measure, _facet_midpoint, _facet_vertices, _u_pattern, _spatial_dim
import ModularEIT: _is_piecewise_constant, _facet_graph, _gram_matrix, _total_variation!, _tv_hessian,
    _tv_gradient_operator, _FacetGraph, _nz_index

export GridapDiscretization

include("Discretization.jl")
include("Assembly.jl")
include("Functions.jl")
include("Boundary.jl")
include("Regularizers.jl")

end # module ModularEITGridap
