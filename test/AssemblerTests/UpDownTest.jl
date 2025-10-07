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
