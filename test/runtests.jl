using ModularEIT
using Test
using Ferrite
using SparseArrays
using LinearAlgebra
using IterativeSolvers


# Just generate grid for all tests
# This is quadrilateral grid
grid = generate_grid(Quadrilateral, (16, 16))
∂Ω = union(getfacetset.((grid,), ["left", "top", "right", "bottom"])...)

order = 3
qr_order = 5

fe = FerriteFESpace{RefQuadrilateral}(grid, order, qr_order, ∂Ω)

@testset "ModularEIT.jl" begin
    # Write your tests here.
    #=
    include("AssemblerTests/BilinearMap.jl")
    include("AssemblerTests/MatrixTests.jl")
    include("AssemblerTests/UpDownTest.jl")
    #include("AssemblerTests/ExportTest.jl")
    include("MeshTests/MeshTests.jl")

    include("SolverTests/SolverTests.jl")
    =#
    include("ReconstructionTests/SimpleImage.jl")
end
