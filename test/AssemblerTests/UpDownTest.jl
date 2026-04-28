# Create test for up and down projector
# Up: boundary coefficients -> force vector
# Down: force vector -> boundary coefficients
using Test
using Ferrite
using SparseArrays
using LinearAlgebra
using IterativeSolvers
using ModularEIT

@testset "UpDownTest" begin
    @testset "Quadrilateral" begin
        # Test up projector
        println(fe.n)
        down = fe.down
        up = fe.up
        # Test down projector

        test_vec = randn(fe.m)
        test_vec2 = randn(fe.n)
        # Test whether up ∘ down == identity
        @test down(up(test_vec)) == test_vec
        @test down(up(down(test_vec2))) == down(test_vec2)
    end
    # Test again for circular mesh from .msh file
    @testset "CustomMesh" begin
        # Test up projector
        println(fe_circ.n)
        down = fe_circ.down
        up = fe_circ.up
        # Test down projector

        test_vec = randn(fe_circ.m)
        test_vec2 = randn(fe_circ.n)
        # Test whether up ∘ down == identity
        @test down(up(test_vec)) == test_vec
        @test down(up(down(test_vec2))) == down(test_vec2)
    end
end
