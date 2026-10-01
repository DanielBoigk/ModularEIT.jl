# Test suite, grouped by component (folders of test/). Run everything, or only some folders:
#     julia --project -e 'using Pkg; Pkg.test()'
#     julia --project -e 'using Pkg; Pkg.test(test_args = ["LinearSolvers", "Data"])'
# The test environment (test/Project.toml) also has the Ferrite back end, which most tests use
# for their discretizations; the back end's own tests are in lib/ModularEITFerrite/test.
using Test

const TESTDIR = @__DIR__
const SELECTED = isempty(ARGS) ? nothing : Set(ARGS)

const SUITES = [
    ("linear solvers", "LinearSolvers", ["test_projected_block_cg.jl", "test_projected_cholesky.jl",
                                         "test_krylov_ldl.jl", "test_gpu_agnostic.jl",
                                         "test_dct_preconditioner.jl", "test_polar_preconditioner.jl"]),
    ("optimization", "Optimization", ["test_regularizers.jl", "test_optimizers.jl", "test_proximal.jl", "test_truncated_svd.jl", "test_confidence.jl", "test_matrix_free.jl"]),
    ("data", "Data", ["test_noise.jl", "test_synthetic.jl", "test_pattern_truncation.jl", "test_approximation_error.jl"]),
]

SELECTED === nothing || issubset(SELECTED, [folder for (_, folder, _) in SUITES]) ||
    error("unknown test folders $(setdiff(SELECTED, [folder for (_, folder, _) in SUITES]))")

@testset "ModularEIT" begin
    for (name, folder, files) in SUITES
        (SELECTED === nothing || folder in SELECTED) || continue
        @testset "$name" begin
            for f in files
                include(joinpath(TESTDIR, folder, f))
            end
        end
    end
end
