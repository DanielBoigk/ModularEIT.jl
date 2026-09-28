using ModularEIT
using LinearAlgebra
using SparseArrays
using Random
using Test

include("helpers.jl")

@testset "Projected block CG" begin
    rng = MersenneTwister(1)
    m = 24
    n = m^2
    A = neumann_laplacian(m; rng)
    B = boundary_currents(m, 8)
    bd = boundary_dofs(m)

    # Reference: minimum-norm solution (orthogonal to the null space).
    Xref = pinv(Matrix(A)) * B

    @testset "mean-zero solution, no preconditioner" begin
        X, stats = pbcg(A, B; rtol = 1e-10)
        @test stats.converged
        @test norm(A * X - B) / norm(B) < 1e-8
        @test maximum(abs, sum(X; dims = 1)) < 1e-8 * norm(X)        # X ⟂ constants
        @test norm(X - Xref) / norm(Xref) < 1e-7
    end

    @testset "boundary-grounded solution" begin
        w = zeros(n); w[bd] .= 1                                      # Σ_{boundary} xᵢ = 0
        X, stats = pbcg(A, B; grounding = w, rtol = 1e-10)
        @test stats.converged
        @test maximum(abs, w' * X) < 1e-8 * norm(X)
        # same solution up to a constant per column
        D = X - Xref
        @test maximum(abs, D .- sum(D; dims = 1) ./ n) < 1e-7 * norm(Xref)
    end

    @testset "general null space of dimension k" begin
        k = 3
        Q = Matrix(qr(randn(rng, 60, 60)).Q)
        λ = [zeros(k); 1 .+ 100 .* rand(rng, 60 - k)]
        A2 = Symmetric(Q * Diagonal(λ) * Q')
        V = Q[:, 1:k] * randn(rng, k, k)                              # non-orthonormal basis is fine
        B2 = A2 * randn(rng, 60, 5)                                   # consistent RHS
        X2, stats = pbcg(A2, B2; nullspace = V, rtol = 1e-12, maxiter = 500)
        @test stats.converged
        @test norm(A2 * X2 - B2) / norm(B2) < 1e-9
        @test norm(Q[:, 1:k]' * X2) < 1e-9 * norm(X2)
    end

    @testset "inconsistent right-hand side is projected" begin
        Bbad = B .+ 0.3                                               # violates Σ bᵢ = 0
        X, stats = pbcg(A, Bbad; rtol = 1e-10)
        @test stats.converged
        @test stats.compatibility_defect > 0.1
        @test norm(A * X - B) / norm(B) < 1e-8                        # solves the projected problem
    end

    @testset "rank-deficient blocks" begin
        Bdep = hcat(B[:, 1], B[:, 1], 2 .* B[:, 2], zeros(n), B[:, 1] + B[:, 2])
        X, stats = pbcg(A, Bdep; rtol = 1e-10)
        @test stats.converged
        @test norm(A * X - Bdep) / norm(Bdep) < 1e-8
        @test norm(X[:, 4]) == 0 || norm(X[:, 4]) < 1e-12
    end

    @testset "single right-hand side" begin
        x, stats = pbcg(A, B[:, 3]; rtol = 1e-10)
        @test x isa Vector
        @test norm(A * x - B[:, 3]) / norm(B[:, 3]) < 1e-8
    end

    @testset "preconditioners reduce iterations" begin
        _, s0 = pbcg(A, B; rtol = 1e-8)
        _, sj = pbcg(A, B; M = JacobiPreconditioner(A), rtol = 1e-8)
        Xa, sa = pbcg(A, B; M = AMGPreconditioner(A, size(B, 2)), rtol = 1e-8)
        @test sj.converged && sa.converged
        @test sj.iterations <= s0.iterations
        @test sa.iterations < s0.iterations ÷ 2
        @test norm(A * Xa - B) / norm(B) < 1e-6
    end

    @testset "AMG V-cycle is a symmetric operator" begin
        P = AMGPreconditioner(A, 2)
        X = randn(rng, n, 2); Y = randn(rng, n, 2)
        PX = similar(X); PY = similar(Y)
        ModularEIT.apply_preconditioner!(PX, P, X)
        ModularEIT.apply_preconditioner!(PY, P, Y)
        @test abs(dot(Y, PX) - dot(X, PY)) < 1e-8 * norm(PX) * norm(Y)
    end

    @testset "preallocated workspace: allocations independent of n" begin
        function allocs(m)
            A = neumann_laplacian(m; rng = MersenneTwister(2))
            B = boundary_currents(m, 4)
            X = zero(B)
            ws = BlockCGWorkspace(A, B)
            pbcg!(X, ws, A, B; maxiter = 20, rtol = 0.0)              # warm up / compile
            fill!(X, 0)
            return @allocated pbcg!(X, ws, A, B; maxiter = 20, rtol = 0.0)
        end
        a_small, a_large = allocs(16), allocs(64)
        @test a_large < a_small + 4096                                # no O(n) allocations
    end

    @testset "Float32" begin
        A32 = Float32.(A); B32 = Float32.(B)
        X32, stats = pbcg(A32, B32; rtol = 1f-5)
        @test eltype(X32) == Float32
        @test stats.converged
        @test norm(A32 * X32 - B32) / norm(B32) < 1f-4
    end
end
