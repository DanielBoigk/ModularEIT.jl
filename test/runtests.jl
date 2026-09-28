using ModularEIT
using Test
using Ferrite, FerriteGmsh
using SparseArrays
using LinearAlgebra
using IterativeSolvers


include("test_projected_block_cg.jl")
include("test_projected_cholesky.jl")
include("test_krylov_ldl.jl")
include("test_gpu_agnostic.jl")
include("test_fem_assembly.jl")
include("test_electrode_models.jl")
include("test_objectives.jl")
include("test_adaptive_meshing.jl")
