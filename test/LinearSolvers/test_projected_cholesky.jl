using ModularEIT
using LinearAlgebra
using SparseArrays
using Random
using Test

isdefined(Main, :neumann_laplacian) || include("helpers.jl")

@testset "Projected Cholesky" begin
    rng = MersenneTwister(4)
    m = 24
    n = m^2
    A = neumann_laplacian(m; rng)
    B = boundary_currents(m, 8)
    bd = boundary_dofs(m)
    Xref = pinv(Matrix(A)) * B                                        # minimum-norm solution

    @testset "mean-zero solution (default grounding)" begin
        F = projected_cholesky(A)
        X = F \ B
        @test norm(A * X - B) / norm(B) < 1e-12
        @test norm(X - Xref) / norm(Xref) < 1e-10
        @test maximum(abs, sum(X; dims = 1)) < 1e-10 * norm(X)
    end

    @testset "boundary grounding, block and single vectors" begin
        w = boundary_grounding(n, bd)
        F = projected_cholesky(A; grounding = w)
        X = F \ B
        @test maximum(abs, w' * X) < 1e-10 * norm(X)
        @test norm(A * X - B) / norm(B) < 1e-12
        for j in axes(B, 2)                                           # vector RHS = block column
            x = F \ B[:, j]
            @test x isa Vector
            @test norm(x - X[:, j]) < 1e-12 * norm(X[:, j])
        end
        # agrees with the projected block CG solver
        Xcg, _ = pbcg(A, B; grounding = w, rtol = 1e-12)
        @test norm(X - Xcg) / norm(X) < 1e-8
    end

    @testset "in-place ldiv! with preallocated output" begin
        F = projected_cholesky(A; nrhs = size(B, 2))
        X = similar(B)
        @test ldiv!(X, F, B) === X
        @test norm(A * X - B) / norm(B) < 1e-12
    end

    @testset "general null space of dimension k" begin
        k = 3
        Q = Matrix(qr(randn(rng, 60, 60)).Q)
        λ = [zeros(k); 1 .+ 100 .* rand(rng, 60 - k)]
        A2 = sparse(Symmetric(Q * Diagonal(λ) * Q'))
        V = Q[:, 1:k] * randn(rng, k, k)
        B2 = A2 * randn(rng, 60, 5)
        F = projected_cholesky(A2; nullspace = V)
        X2 = F \ B2
        @test norm(A2 * X2 - B2) / norm(B2) < 1e-9
        @test norm(Q[:, 1:k]' * X2) < 1e-9 * norm(X2)
    end

    @testset "inconsistent right-hand side is projected" begin
        F = projected_cholesky(A)
        X = F \ (B .+ 0.3)
        @test norm(A * X - B) / norm(B) < 1e-12
    end

    @testset "refactorisation for a new conductivity (same pattern)" begin
        F = projected_cholesky(A; grounding = boundary_grounding(n, bd))
        A2 = neumann_laplacian(m; rng = MersenneTwister(99))           # same sparsity pattern
        @test A2.colptr == A.colptr && A2.rowval == A.rowval
        refactor!(F, A2)
        X = F \ B
        @test norm(A2 * X - B) / norm(B) < 1e-12
    end

    @testset "ldiv! allocates only CHOLMOD's solve workspace" begin
        # CHOLMOD (via SparseArrays) allocates its nJ × s solve workspace on every call;
        # everything else in ldiv! is allocation-free.
        function allocs(m)
            A = neumann_laplacian(m; rng = MersenneTwister(2))
            B = boundary_currents(m, 4)
            F = projected_cholesky(A; nrhs = 4)
            X = similar(B)
            ldiv!(X, F, B)
            return (@allocated ldiv!(X, F, B)) - (@allocated ldiv!(F.xJ, F.fact, F.bJ))
        end
        @test allocs(64) < allocs(16) + 4096
    end

    @testset "CHOLMOD workspace is reused (no allocation per solve)" begin
        # every solve used to allocate an n × s workspace in CHOLMOD through Julia's counting
        # allocator without its release being credited, so the GC heuristics drifted and real
        # garbage piled up; the workspace is now kept with the factorisation
        n = 2000
        A = spdiagm(-1 => -ones(n - 1), 0 => 2.1 .* ones(n), 1 => -ones(n - 1))
        F = projected_cholesky(A)
        B = randn(n, 64)
        X = similar(B)
        ldiv!(X, F, B)
        @test X ≈ projected_ldl(A) \ B rtol = 1e-8                 # independent backend
        live0 = Base.gc_live_bytes()
        for _ in 1:100
            ldiv!(X, F, B)
        end
        @test Base.gc_live_bytes() - live0 < 20 * n * 64 * 8      # (was 100 workspaces)
        @test X ≈ projected_ldl(A) \ B rtol = 1e-8                 # reused workspace, same result
    end
end
