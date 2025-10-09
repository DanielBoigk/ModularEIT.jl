# Here we test whether the stiffness matrix is correctly computed.

using Test


using Ferrite
using SparseArrays
using LinearAlgebra
using IterativeSolvers
using ModularEIT
@testset "Matrix Tests" begin
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
    @test Matrix_norm < 10.0

    # Implement a sanity check if the two matrices assembled from the function and the vector are roughly the same (use relatively coarse ≈ )
    Matrix_norm = norm(KD_vec - KD_func)
    println("Norm of Matrix difference: ", Matrix_norm)
    @test Matrix_norm < 10.0

end
