# Test suite, grouped by component. Every file can also be run on its own, e.g.
#     julia --project -e 'using Test; include("test/Galerkin/test_objectives.jl")'
using Test

const TESTDIR = @__DIR__
runfolder(folder, files) = for f in files
    include(joinpath(TESTDIR, folder, f))
end

@testset "ModularEIT" begin
    @testset "linear solvers" begin
        runfolder("LinearSolvers", ["test_projected_block_cg.jl", "test_projected_cholesky.jl",
                                    "test_krylov_ldl.jl", "test_gpu_agnostic.jl"])
    end
    @testset "Galerkin layer" begin
        runfolder("Galerkin", ["test_fem_assembly.jl", "test_electrode_models.jl",
                               "test_objectives.jl", "test_pattern_svd.jl"])
    end
    @testset "adaptivity" begin
        runfolder("Adaptivity", ["test_adaptive_meshing.jl", "test_bisection.jl",
                                 "test_residual_estimator.jl"])
    end
    @testset "optimization" begin
        runfolder("Optimization", ["test_regularizers.jl", "test_optimizers.jl"])
    end
    @testset "images" begin
        runfolder("Images", ["test_images.jl"])
    end
end
