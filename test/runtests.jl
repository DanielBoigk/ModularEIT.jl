using ModularEIT
using Test
using Ferrite, FerriteGmsh
using SparseArrays
using LinearAlgebra
using IterativeSolvers

@testset "ModularEIT.jl" begin
    mesh = circle_mesh(16)
    @test nnodes(mesh) == 17
    @test nelements(mesh) == 16
    @test length(ring_electrodes(mesh, 4)) == 4
    @test penalty(Tikhonov(2.0), [1.0, 2.0]) == 5.0
    prob = ForwardProblem(mesh, ring_electrodes(mesh, 4))
    @test sum(solve_forward(prob, ones(16), [1.0, 0.0, -1.0, 0.0])) ≈ 0 atol = 1e-12
end

include("test_projected_block_cg.jl")
include("test_projected_cholesky.jl")
