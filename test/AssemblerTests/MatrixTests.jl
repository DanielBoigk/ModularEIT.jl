# Here we test whether the stiffness matrix is correctly computed.

using Test


using Ferrite
using SparseArrays
using LinearAlgebra
using IterativeSolvers
using ModularEIT

grid = generate_grid(Quadrilateral, (16, 16));
∂Ω = union(getfacetset.((grid,), ["left", "top", "right", "bottom"])...)

order = 3
qr_order = 5

fe = FerriteFESpace{RefQuadrilateral}(grid, order, qr_order, ∂Ω)

#we just take some sample conductivity function:
conductivity = (x) -> 1.1 + sin(x[1]) * cos(x[2])

cond_vec = project_function_to_fem(fe, conductivity)

KN_func = assemble_L(fe, conductivity)
KN_vec = assemble_L(fe, cond_vec)

KD_func = to_dirichlet(KN_func, fe)
KD_vec = to_dirichlet(KN_vec, fe)

# Implement a sanity check if the two matrices assembled from the function and the vector are roughly the same (use relatively coarse ≈ )
Matrix_norm = norm(KN_vec - KN_func)
println("Norm of Matrix difference: ", Matrix_norm)
@assert Matrix_norm < 10.0

# Implement a sanity check if the two matrices assembled from the function and the vector are roughly the same (use relatively coarse ≈ )
Matrix_norm = norm(KD_vec - KD_func)
println("Norm of Matrix difference: ", Matrix_norm)
@assert Matrix_norm < 10.0


# Make sanity check if Dirichlet and Neumann boundary operators are compatible with up and down projections.


# Load gmsh grid and repeat test.
u = randn(fe.n)


diff_u = KD_func * u - KN_func * u
num_approx_zero = 0
for i in 1:fe.n
    if abs(diff_u[i]) < 1e-10
        num_approx_zero += 1
    end
end
println("Number of approx. zero entries: ", num_approx_zero)
println("desired Number of approx. zero entries: ", fe.m)
@assert num_approx_zero == fe.m
