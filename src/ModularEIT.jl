"""
    ModularEIT

Modular building blocks for Electrical Impedance Tomography (EIT): finite element
discretizations, electrode models, forward solvers, adjoint-state and Kohn–Vogelius objectives
and projected linear solvers that can be combined freely.
"""
module ModularEIT

using LinearAlgebra


export BlockCGWorkspace, BlockCGStats, pbcg, pbcg!, boundary_grounding
export JacobiPreconditioner, AMGPreconditioner
export ProjectedCholesky, projected_cholesky, projected_ldl, refactor!
export ProjectedMinresWorkspace, BlockMinresStats, pbminres, pbminres!
export DeviceSparseMatrixCSR, device_converter
export AbstractLinearSolver, DirectSolver, BlockCGSolver
export StructuredGrid, DCTPreconditioner, dct_preconditioner
export AbstractFastPreconditioner, PolarStructure, PolarPreconditioner, polar_preconditioner, polar_grid
export ConformalMap, map_derivative, ConformalGrid, conformal_grid

export AbstractDiscretization, AbstractElectrodeModel, AbstractForwardModel, AbstractObjective
export AbstractMisfit, AbstractRieszMap, AbstractEITProblem, AbstractSolutionState
export AbstractRegularizer, AbstractOptimizer

export FerriteDiscretization, ndofs_u, ndofs_σ
export FEMatrices, assemble_mass, assemble_mass!, assemble_stiffness, assemble_stiffness!
export assemble_boundary_mass, assemble_boundary_mass!, assemble_boundary_load!
export assemble_weighted_stiffness, assemble_weighted_stiffness!, weighted_stiffness_values!
export ConductivityTensor, pair_products!, tensor_gradient!
export CoefficientGradient, L2Gradient, riesz_map, riesz_map!
export interpolate_function, l2_project, fe_inner, fe_norm, total_variation, total_variation!

export ContinuumModel, PointElectrodeModel, GapModel, CompleteElectrodeModel
export angular_electrodes, electrode_length, transfer_electrodes
export ForwardModel, system_matrix!, n_inject, n_measure, n_control, trigonometric_patterns
export forward_neumann, forward_dirichlet, reground, pattern_svd
export ImageMap, to_image, from_image, UnitImage, unit_image, from_unit_image
export AdaptiveMesh, current_grid, refine_mesh!, coarsen_mesh!, is_nonconforming, cell_levels, max_level
export residual_indicator, flux_recovery_indicator, goal_oriented_indicator, jump_indicator, dorfler_marking, transfer_conductivity
export SquaredEuclidean, WeightedSquaredEuclidean
export AdjointStateObjective, KohnVogeliusObjective, objective_value, value_and_gradient!
export residual!, residual_and_jacobian!, n_residual, boundary_error, pattern_values
export TikhonovRegularizer, TotalVariationRegularizer, RegularizedObjective, gauss_newton_hessian
export minimize, OptimizationState, GradientDescent, LBFGS, GaussNewton
export prox, prox!, ProximalMap, lumped_mass, ProximalGradient, ADMM
export AbstractNoiseModel, GaussianNoise, RelativeGaussianNoise, SourceMeterNoise, add_noise, add_noise!
export expected_squared_error, discrepancy_target, perturb_boundary_operator, perturb_contact_impedance, electrode_angles
export AbstractInclusion, CircleInclusion, EllipseInclusion, PolygonInclusion, InclusionPhantom, random_inclusions
export PixelFunction, image_phantom, TransformedPhantom, lognormal_phantom, levelset_phantom
export gaussian_random_field, corrupt_image, conductivity, simulate_data


include("AbstractTypes.jl")
include("Geometry/ConformalMap.jl")

include("LinearSolvers/ProjectedBlockCG.jl")
include("LinearSolvers/ProjectedCholesky.jl")
include("LinearSolvers/DeviceSparse.jl")
include("LinearSolvers/ProjectedBlockMinres.jl")
include("LinearSolvers/SolverInterface.jl")
include("LinearSolvers/DCT.jl")
include("LinearSolvers/Polar.jl")

include("Galerkin/ForwardModel.jl")
include("Galerkin/Regularizers.jl")
include("Galerkin/Ferrite/Ferrite.jl")
include("Galerkin/FastPreconditioner.jl")
include("Galerkin/Objectives/Misfits.jl")
include("Galerkin/Objectives/AdjointState.jl")
include("Galerkin/Objectives/KohnVogelius.jl")

include("Optimization/Optimizers.jl")
include("Optimization/FirstOrder.jl")
include("Optimization/GaussNewton.jl")
include("Optimization/Proximal.jl")

include("Data/Noise.jl")
include("Data/Phantoms.jl")
include("Data/Simulation.jl")

end # module ModularEIT
