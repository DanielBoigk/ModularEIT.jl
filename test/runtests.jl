using ModularEIT
using Test
using Ferrite, FerriteGmsh
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

# Load from .msh file and mesh with Triangular
grid_circ = togrid("circle.msh")
∂Ω_circ = union(getfacetset.((grid_circ,), ["boundary"])...)
fe_circ = FerriteFESpace{RefTriangle}(grid_circ, order, qr_order, ∂Ω_circ)

# Load from .msh file and mesh with Triangular

@testset "ModularEIT.jl" begin
    # Write your tests here.

    include("AssemblerTests/BilinearMap.jl")
    include("AssemblerTests/MatrixTests.jl")
    include("AssemblerTests/UpDownTest.jl")
    include("AssemblerTests/SVDTest.jl")
    # #include("AssemblerTests/ExportTest.jl")
    # include("MeshTests/MeshTests.jl")

    # include("SolverTests/SolverTests.jl")
    # include("OptimizerTests/LBFGS_BasicTest.jl")
    # include("GradientTests/NeumannAdjointGradientTest.jl")
    # include("ReconstructionTests/ReconstructionTests.jl")
end
