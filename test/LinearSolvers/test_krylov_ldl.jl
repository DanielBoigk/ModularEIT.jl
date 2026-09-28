using ModularEIT
using LinearAlgebra
using SparseArrays
using Random
using Test

isdefined(Main, :neumann_laplacian) || include("helpers.jl")

@testset "Projected LDLᵀ (LDLFactorizations.jl)" begin
    m = 24
    n = m^2
    A = neumann_laplacian(m; rng = MersenneTwister(5))
    B = boundary_currents(m, 6)
    bd = boundary_dofs(m)
    w = boundary_grounding(n, bd)

    @testset "Float64: boundary grounding, agrees with CHOLMOD" begin
        F = projected_ldl(A; grounding = w)
        X = F \ B
        @test norm(A * X - B) / norm(B) < 1e-12
        @test maximum(abs, w' * X) < 1e-10 * norm(X)
        Xc = projected_cholesky(A; grounding = w) \ B
        @test norm(X - Xc) / norm(Xc) < 1e-12
    end

    @testset "vector and block right-hand sides" begin
        F = projected_ldl(A; grounding = w, nrhs = size(B, 2))
        X = similar(B)
        ldiv!(X, F, B)
        for j in axes(B, 2)
            x = F \ B[:, j]
            @test x isa Vector
            @test norm(x - X[:, j]) < 1e-12 * norm(X[:, j])
        end
    end

    @testset "Float32 (not supported by CHOLMOD)" begin
        A32, B32 = Float32.(A), Float32.(B)
        F = projected_ldl(A32; grounding = Float32.(w))
        X = F \ B32
        @test eltype(X) == Float32
        @test norm(A32 * X - B32) / norm(B32) < 1f-4
        @test maximum(abs, Float32.(w)' * X) < 1f-4 * norm(X)
    end

    @testset "refactorisation for a new conductivity" begin
        F = projected_ldl(A; grounding = w)
        A2 = neumann_laplacian(m; rng = MersenneTwister(123))
        refactor!(F, A2)
        X = F \ B
        @test norm(A2 * X - B) / norm(B) < 1e-12
    end

    @testset "inconsistent data and general null space" begin
        F = projected_ldl(A)
        @test norm(A * (F \ (B .+ 1)) - B) / norm(B) < 1e-12
        rng = MersenneTwister(6)
        Q = Matrix(qr(randn(rng, 50, 50)).Q)
        A3 = sparse(Symmetric(Q * Diagonal([0; 0; 1 .+ rand(rng, 48)]) * Q'))
        B3 = A3 * randn(rng, 50, 3)
        X3 = projected_ldl(A3; nullspace = Q[:, 1:2]) \ B3
        @test norm(A3 * X3 - B3) / norm(B3) < 1e-9
    end
end

@testset "Projected block MINRES (Krylov.jl)" begin
    m = 24
    n = m^2
    A = neumann_laplacian(m; rng = MersenneTwister(7))
    B = boundary_currents(m, 6)
    bd = boundary_dofs(m)
    w = boundary_grounding(n, bd)
    Xref = projected_cholesky(A; grounding = w) \ B

    for scaling in (:none, :jacobi)
        @testset "boundary grounding, scaling = $scaling" begin
            X, stats = pbminres(A, B; grounding = w, scaling, rtol = 1e-10)
            @test stats.converged
            @test norm(A * X - B) / norm(B) < 1e-8
            @test maximum(abs, w' * X) < 1e-8 * norm(X)
            @test norm(X - Xref) / norm(Xref) < 1e-7
        end
    end

    @testset "columnwise mode (single-vector MINRES) agrees" begin
        X, stats = pbminres(A, B; grounding = w, columnwise = true, rtol = 1e-10)
        @test stats.converged
        @test norm(X - Xref) / norm(Xref) < 1e-7
    end

    @testset "single right-hand side" begin
        x, stats = pbminres(A, B[:, 2]; grounding = w, rtol = 1e-10)
        @test x isa Vector
        @test norm(x - Xref[:, 2]) / norm(Xref[:, 2]) < 1e-7
    end

    @testset "preallocated workspace, repeated solves" begin
        ws = ProjectedMinresWorkspace(A, B; grounding = w)
        X = similar(B)
        for k in 1:2
            stats = pbminres!(X, ws, A, B; rtol = 1e-10)
            @test stats.converged
            @test norm(X - Xref) / norm(Xref) < 1e-7
        end
    end

    @testset "inconsistent right-hand side is projected" begin
        X, stats = pbminres(A, B .+ 0.5; grounding = w, rtol = 1e-10)
        @test stats.compatibility_defect > 0.1
        @test norm(X - Xref) / norm(Xref) < 1e-7
    end

    @testset "rank-deficient block" begin
        Bdep = hcat(B[:, 1], B[:, 1], zeros(n), 2 .* B[:, 3])
        X, stats = pbminres(A, Bdep; grounding = w, rtol = 1e-10)
        @test stats.converged
        @test norm(A * X - Bdep) / norm(Bdep) < 1e-8
    end

    @testset "Float32" begin
        X, stats = pbminres(Float32.(A), Float32.(B); grounding = Float32.(w), rtol = 1f-5)
        @test eltype(X) == Float32
        @test norm(Float32.(A) * X - Float32.(B)) / norm(B) < 1f-3
    end
end
