# Backend-agnostic GPU code paths, tested with JLArrays.jl: a CPU implementation of the GPUArrays
# interface with scalar indexing disabled. Anything that passes here only uses operations every
# GPUArrays/KernelAbstractions backend (CUDA, AMDGPU, oneAPI, Metal) provides.
using ModularEIT
using ModularEITFerrite
using LinearAlgebra
using SparseArrays
using Random
using Test
using JLArrays
import Ferrite

isdefined(Main, :neumann_laplacian) || include("helpers.jl")

JLArrays.allowscalar(false)

@testset "GPU-agnostic code paths (JLArrays)" begin
    m = 20
    n = m^2
    A = neumann_laplacian(m; rng = MersenneTwister(11))
    B = boundary_currents(m, 5)
    bd = boundary_dofs(m)
    w = boundary_grounding(n, bd)
    Xref = projected_cholesky(A; grounding = w) \ B
    to_device = device_converter(JLArray)

    @testset "generic device CSR matrix: sparse × dense" begin
        Ad = to_device(A)
        @test Ad isa DeviceSparseMatrixCSR
        Xd = JLArray(B)
        @test Array(Ad * Xd) ≈ A * B
        @test Array(Ad * JLArray(B[:, 1])) ≈ A * B[:, 1]
        Y = JLArray(ones(n, 5))
        mul!(Y, Ad, Xd, 2.0, 0.5)
        @test Array(Y) ≈ 2 .* (A * B) .+ 0.5
        # non-symmetric matrix (e.g. AMG prolongation)
        P = sprand(MersenneTwister(1), n, 30, 0.1)
        @test Array(to_device(P) * JLArray(randn(MersenneTwister(2), 30, 3))) ≈ P * randn(MersenneTwister(2), 30, 3)
    end

    for pre in (:none, :jacobi, :amg)
        @testset "projected block CG, preconditioner = $pre" begin
            M = pre == :none ? nothing :
                pre == :jacobi ? JacobiPreconditioner(A; to_device) :
                                 AMGPreconditioner(A, size(B, 2); to_device)
            X, stats = pbcg(to_device(A), JLArray(B); M, grounding = w, rtol = 1e-10)
            @test X isa JLArray
            @test stats.converged
            @test norm(Array(X) - Xref) / norm(Xref) < 1e-7
        end
    end

    @testset "projected factorisation, host fallback" begin
        F = projected_cholesky(A; grounding = w, nrhs = size(B, 2), to_device)
        X = F \ JLArray(B)
        @test X isa JLArray
        @test norm(Array(X) - Xref) / norm(Xref) < 1e-10
        x = F \ JLArray(B[:, 2])
        @test norm(Array(x) - Xref[:, 2]) / norm(Xref[:, 2]) < 1e-10
        A2 = neumann_laplacian(m; rng = MersenneTwister(12))
        refactor!(F, A2)
        @test norm(A2 * Array(F \ JLArray(B)) - B) / norm(B) < 1e-10
        F32 = projected_cholesky(Float32.(A); grounding = Float32.(w), to_device)   # LDLᵀ on the host
        @test norm(Array(F32 \ JLArray(Float32.(B))) - Xref) / norm(Xref) < 1e-3
    end

    @testset "projected MINRES (columnwise)" begin
        X, stats = pbminres(to_device(A), JLArray(B); grounding = w, diagonal = Vector(diag(A)),
                            columnwise = true, rtol = 1e-10)
        @test X isa JLArray
        @test stats.converged
        @test norm(Array(X) - Xref) / norm(Xref) < 1e-7
    end

    @testset "conductivity tensor: assembly and gradient contraction on the device" begin
        grid = Ferrite.generate_grid(Ferrite.Triangle, (6, 6))
        disc = FerriteDiscretization(grid; ip_σ = Ferrite.Lagrange{Ferrite.RefTriangle, 1}())
        ct = ConductivityTensor(disc)
        ctd = ConductivityTensor(disc; to_device)
        @test ctd.T isa DeviceSparseMatrixCSR
        rng = MersenneTwister(4)
        σ = 1 .+ rand(rng, ndofs_σ(disc))
        nz = weighted_stiffness_values!(zeros(nnz(ct.pattern)), ct, σ)
        nzd = weighted_stiffness_values!(JLArray(zeros(nnz(ct.pattern))), ctd, JLArray(σ))
        @test nzd isa JLArray
        @test Array(nzd) ≈ nz
        Λ, U = randn(rng, ndofs_u(disc), 3), randn(rng, ndofs_u(disc), 3)
        g = tensor_gradient!(zeros(ndofs_σ(disc)), ct, Λ, U; α = -1)
        gd = tensor_gradient!(JLArray(zeros(ndofs_σ(disc))), ctd, JLArray(Λ), JLArray(U); α = -1)
        @test Array(gd) ≈ g
    end
end
