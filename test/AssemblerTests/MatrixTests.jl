# Here we test whether the stiffness matrix is correctly computed.

using Test


using Ferrite
using SparseArrays
using LinearAlgebra
using IterativeSolvers
using ModularEIT

@testset "Boundary Operators" begin
    MΓ, KΓ, Hn½, H½ = assemble_boundary_matrices(fe)
    H½inv = inv(H½)
    Mb  = MΓ |> Matrix |> Symmetric
    IZ = Mb * H½inv * Mb - Hn½
    @test (maximum(IZ) < 1e-10) && (minimum(IZ) > -1e-10) # wish it would have better tolerances

    # write some more tests for SVD later

end

@testset "Matrix Tests" begin
    #we just take some sample conductivity function:
    conductivity = (x) -> 1.1 + sin(x[1]) * cos(x[2])
    @testset "Quadrilateral" begin
    # project the conductivity function to the finite element space
    cond_vec = project_function_to_fem(fe, conductivity)

    # assemble the stiffness matrix from the conductivity function
    KN_func = assemble_L(fe, conductivity)
    KN_vec = assemble_L(fe, cond_vec)
    # convert to Dirichlet matrix
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

    @testset "Quadrilateral" begin
        # project the conductivity function to the finite element space
        cond_vec = project_function_to_fem(fe_circ, conductivity)

        # assemble the stiffness matrix from the conductivity function
        KN_func = assemble_L(fe_circ, conductivity)
        KN_vec = assemble_L(fe_circ, cond_vec)
        # convert to Dirichlet matrix
        KD_func = to_dirichlet(KN_func, fe_circ)
        KD_vec = to_dirichlet(KN_vec, fe_circ)

        # Implement a sanity check if the two matrices assembled from the function and the vector are roughly the same (use relatively coarse ≈ )
        Matrix_norm = norm(KN_vec - KN_func)
        println("Norm of Matrix difference: ", Matrix_norm)
        @test Matrix_norm < 10.0

        # Implement a sanity check if the two matrices assembled from the function and the vector are roughly the same (use relatively coarse ≈ )
        Matrix_norm = norm(KD_vec - KD_func)
        println("Norm of Matrix difference: ", Matrix_norm)
        @test Matrix_norm < 10.0
        
    end
    # Define boundary
end
