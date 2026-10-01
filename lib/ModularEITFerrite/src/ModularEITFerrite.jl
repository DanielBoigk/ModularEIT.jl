"""
    ModularEITFerrite

[Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl) back end of ModularEIT: finite element
discretizations with separate spaces for the potential and the conductivity, assembly, electrode
geometry and forward models, grids (structured, polar, conformally mapped), image maps, pixel
parametrizations, total variation on meshes, and adaptive meshing (hanging nodes on
quadrilaterals, newest vertex bisection on triangles).
"""
module ModularEITFerrite

using LinearAlgebra
using SparseArrays
using Ferrite
using ModularEIT

# generic functions and types of ModularEIT that this back end extends or uses
import ModularEIT: ADMM, AbstractDiscretization, AbstractLinearSolver, AbstractParametrization,
    AbstractRegularizer, CompleteElectrodeModel, ConductivityTensor, ConformalMap, ContinuumModel,
    DirectSolver, FEMatrices, ForwardModel, GapModel, ParametrizedObjective, PointElectrodeModel,
    PolarPreconditioner, PolarStructure, ProximalGradient, StructuredGrid, SubspaceParametrization,
    TikhonovRegularizer, TotalVariationRegularizer, angular_electrodes, assemble_weighted_stiffness,
    assemble_weighted_stiffness!, boundary_grounding, conductivity, dorfler_marking,
    electrode_angles, electrode_length, fe_inner, fe_norm, forward_dirichlet, forward_neumann,
    gauss_newton_hessian, interpolate_function, l2_project, lumped_mass, n_measure, ndofs_u, ndofs_σ,
    objective_value, pixel_image, polar_structure, prox, prox!, structured_grid, system_matrix!,
    total_variation, total_variation!, transfer_electrodes, value_and_gradient!
import ModularEIT: _as_matrix, _bilinear, _dct_matrix, _forward_model, _init_neumann_solver,
    _nz_index, _pixel_value, _project_adjoint!, _reference_structure, _solve!
# electrode primitives (back end contract) and electrode helpers
import ModularEIT: _boundary_facets, _boundary_dofs, _boundary_mass, _boundary_load, _facet_free_dofs,
    _facet_measure, _facet_midpoint, _facet_vertices, _u_pattern, _spatial_dim, _dof_coordinates,
    _centroid, _angles, _electrode_angle, _boundary_weights, _grounding, _electrode_averages,
    _nearest_boundary_dofs, _segment_distance
# regularizer primitives
import ModularEIT: _is_piecewise_constant, _facet_graph, _gram_matrix, _total_variation!, _tv_hessian,
    _tv_gradient_operator, _FacetGraph

export FerriteDiscretization, is_nonconforming
export assemble_mass, assemble_mass!, assemble_stiffness, assemble_stiffness!
export assemble_boundary_mass, assemble_boundary_mass!, assemble_boundary_load!
export polar_grid, ConformalGrid, conformal_grid
export ImageMap, to_image, from_image, UnitImage, unit_image, from_unit_image
export PixelParametrization, pixel_parameters, dct_basis, boundary_band_basis
export AdaptiveMesh, current_grid, refine_mesh!, coarsen_mesh!, cell_levels, max_level
export residual_indicator, flux_recovery_indicator, goal_oriented_indicator, jump_indicator, transfer_conductivity

include("Discretization.jl")
include("Assemblers/MatrixAssemblers.jl")
include("Assemblers/TensorAssembler.jl")
include("Assemblers/CoeffAssembler.jl")
include("FESpace.jl")
include("StructuredGrid.jl")
include("PolarGrid.jl")
include("Regularizers.jl")
include("Boundary.jl")
include("Images.jl")
include("Parametrization.jl")
include("AdaptiveMeshing.jl")
include("ResidualEstimator.jl")

end # module ModularEITFerrite
