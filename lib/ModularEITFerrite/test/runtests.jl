# Tests of the Ferrite back end, grouped by component (folders of test/). Run everything, or only
# some folders (from lib/ModularEITFerrite):
#     julia --project -e 'using Pkg; Pkg.test()'
#     julia --project -e 'using Pkg; Pkg.test(test_args = ["Images", "Adaptivity"])'

using Test

const TESTDIR = @__DIR__
const SELECTED = isempty(ARGS) ? nothing : Set(ARGS)

const SUITES = [
    ("Galerkin layer", "Galerkin", ["test_fem_assembly.jl", "test_electrode_models.jl",
                                    "test_objectives.jl", "test_pattern_svd.jl", "test_conformal.jl",
                                    "test_parametrization.jl"]),
    ("adaptivity", "Adaptivity", ["test_adaptive_meshing.jl", "test_bisection.jl", "test_residual_estimator.jl"]),
    ("images", "Images", ["test_images.jl"]),
]

SELECTED === nothing || issubset(SELECTED, [folder for (_, folder, _) in SUITES]) ||
    error("unknown test folders $(setdiff(SELECTED, [folder for (_, folder, _) in SUITES]))")

@testset "ModularEITFerrite" begin
    for (name, folder, files) in SUITES
        (SELECTED === nothing || folder in SELECTED) || continue
        @testset "$name" begin
            for f in files
                include(joinpath(TESTDIR, folder, f))
            end
        end
    end
end
