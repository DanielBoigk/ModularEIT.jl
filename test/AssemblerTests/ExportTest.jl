using Interpolations
using LinearAlgebra
using Test


@testset
grid = generate_grid(Quadrilateral, (16, 16))
∂Ω = union(getfacetset.((grid,), ["left", "top", "right", "bottom"])...)

order = 3
qr_order = 5

fe = FerriteFESpace{RefQuadrilateral}(grid, order, qr_order, ∂Ω)



end
