"""
    ModularEIT

Modular building blocks for Electrical Impedance Tomography (EIT): electrode models, forward
solvers, adjoint-state and Kohn–Vogelius objectives, regularizers, optimizers and projected linear
solvers that can be combined freely. The finite element discretization comes from a back end
package: ModularEITFerrite (Ferrite.jl) or ModularEITGridap (Gridap.jl).
"""
module ModularEIT

using LinearAlgebra


export BlockCGWorkspace, BlockCGStats, pbcg, pbcg!, boundary_grounding
export JacobiPreconditioner, AMGPreconditioner
export ProjectedCholesky, projected_cholesky, projected_ldl, refactor!
export ProjectedMinresWorkspace, BlockMinresStats, pbminres, pbminres!
export DeviceSparseMatrixCSR, device_converter
export AbstractLinearSolver, DirectSolver, BlockCGSolver, InfeasibleConductivityError
export StructuredGrid, DCTPreconditioner, dct_preconditioner
export AbstractFastPreconditioner, PolarStructure, PolarPreconditioner, polar_preconditioner
export ConformalMap, map_derivative

export AbstractDiscretization, AbstractElectrodeModel, AbstractForwardModel, AbstractObjective
export AbstractMisfit, AbstractRieszMap, AbstractEITProblem, AbstractSolutionState
export AbstractRegularizer, AbstractOptimizer

# back end contract (methods are added by ModularEITFerrite, ModularEITGridap)
export ndofs_u, ndofs_σ, interpolate_function, l2_project, fe_inner, fe_norm, total_variation, total_variation!
export lumped_mass, angular_electrodes, electrode_length, transfer_electrodes, structured_grid, polar_structure
export assemble_weighted_stiffness, pixel_image

export FEMatrices, ConductivityTensor, assemble_weighted_stiffness!, weighted_stiffness_values!
export pair_products!, tensor_gradient!
export CoefficientGradient, L2Gradient, riesz_map, riesz_map!

export ContinuumModel, PointElectrodeModel, GapModel, CompleteElectrodeModel
export ForwardModel, system_matrix!, n_inject, n_measure, n_control, trigonometric_patterns
export forward_neumann, forward_dirichlet, reground, pattern_svd, truncate_patterns
export dorfler_marking
export SquaredEuclidean, WeightedSquaredEuclidean, ProjectedMisfit
export AdjointStateObjective, KohnVogeliusObjective, objective_value, value_and_gradient!
export residual, residual!, residual_and_jacobian!, n_residual, boundary_error, pattern_values
export jacobian_operator, JacobianOperator, ParametrizedJacobian, jacobian_column_norms, jacobian_gram
export TikhonovRegularizer, TotalVariationRegularizer, RegularizedObjective, gauss_newton_hessian
export minimize, OptimizationState, GradientDescent, LBFGS, GaussNewton
export jacobian_svd, jacobian_basis, TruncatedGaussNewton
export sensitivity_map, resolution_map, posterior_std
export prox, prox!, ProximalMap, ProximalGradient, ADMM
export AbstractNoiseModel, GaussianNoise, RelativeGaussianNoise, SourceMeterNoise, add_noise, add_noise!
export ApproximationError, whiten, ApproximationErrorObjective
export expected_squared_error, discrepancy_target, perturb_boundary_operator, perturb_contact_impedance, electrode_angles
export AbstractInclusion, CircleInclusion, EllipseInclusion, PolygonInclusion, InclusionPhantom, random_inclusions
export PixelFunction, image_phantom, TransformedPhantom, lognormal_phantom, levelset_phantom
export gaussian_random_field, corrupt_image, conductivity, simulate_data
export AbstractParametrization, SubspaceParametrization, ParametrizedObjective, parameter_count
export EITProblem, reconstruct!, solution, data_misfit


include("AbstractTypes.jl")
include("Geometry/ConformalMap.jl")

include("LinearSolvers/Projection.jl")
include("LinearSolvers/ProjectedBlockCG.jl")
include("LinearSolvers/ProjectedCholesky.jl")
include("LinearSolvers/DeviceSparse.jl")
include("LinearSolvers/ProjectedBlockMinres.jl")
include("LinearSolvers/SolverInterface.jl")
include("LinearSolvers/DCT.jl")
include("LinearSolvers/Polar.jl")

include("Galerkin/Discretization.jl")
include("Galerkin/ElectrodeModels.jl")
include("Galerkin/ForwardModel.jl")
include("Galerkin/Electrodes.jl")
include("Galerkin/PatternSVD.jl")
include("Galerkin/Marking.jl")
include("Galerkin/Regularizers.jl")
include("Galerkin/RegularizersFE.jl")
include("Galerkin/FastPreconditioner.jl")
include("Galerkin/Objectives/Misfits.jl")
include("Galerkin/Objectives/AdjointState.jl")
include("Galerkin/Objectives/KohnVogelius.jl")
include("Galerkin/Parametrization.jl")
include("Galerkin/JacobianBlocks.jl")

include("Optimization/Optimizers.jl")
include("Optimization/FirstOrder.jl")
include("Optimization/GaussNewton.jl")
include("Optimization/TruncatedSVD.jl")
include("Optimization/Confidence.jl")
include("Optimization/Proximal.jl")

include("Data/Noise.jl")
include("Data/ApproximationError.jl")
include("Data/Phantoms.jl")
include("Data/Simulation.jl")

include("Reconstruction/Problem.jl")

end # module ModularEIT
