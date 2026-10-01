# Tests of the Gridap back end: on identical meshes it must reproduce the Ferrite back end
# (matrices up to the dof numbering, voltages of all electrode models, objective values and
# gradients, regularizers), and work on Gridap's own models.
#     julia --project=lib/ModularEITGridap -e 'using Pkg; Pkg.test()'
using Test

@testset "ModularEITGridap" begin
    include("test_against_ferrite.jl")
    include("test_gridap_models.jl")
end
