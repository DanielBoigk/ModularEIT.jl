# Finite element layer: discretization (two DofHandlers), matrix assembly, the conductivity
# tensor T with nzval(L(σ)) = T σ, gradient contractions, Riesz maps, coefficient assembly and
# FE-space functionals. Meshes: triangles and quadrilaterals on [-1, 1]².
using ModularEIT
using ModularEITFerrite
using Ferrite
using SparseArrays
using LinearAlgebra
using Random
using Test

const SQUARE_AREA = 4.0
const SQUARE_PERIMETER = 8.0

# (grid, ip_u, ip_σ) combinations: P1/P0, P1/P1, P2/P1 on triangles, Q1/Q0, Q2/Q1 on quads
function fem_cases(n = 5)
    tri, quad = generate_grid(Triangle, (n, n)), generate_grid(Quadrilateral, (n, n))
    return [
        ("P1/P0", tri, Lagrange{RefTriangle, 1}(), DiscontinuousLagrange{RefTriangle, 0}()),
        ("P1/P1", tri, Lagrange{RefTriangle, 1}(), Lagrange{RefTriangle, 1}()),
        ("P2/P1", tri, Lagrange{RefTriangle, 2}(), Lagrange{RefTriangle, 1}()),
        ("Q1/Q0", quad, Lagrange{RefQuadrilateral, 1}(), DiscontinuousLagrange{RefQuadrilateral, 0}()),
        ("Q2/Q1", quad, Lagrange{RefQuadrilateral, 2}(), Lagrange{RefQuadrilateral, 1}()),
    ]
end

# nodal/interpolated coefficient vector of x ↦ f(x) in the u space
coeffs_u(disc, f) = interpolate_function(disc, f; field = :u)

@testset "FE assembly" begin
    rng = MersenneTwister(3)

    @testset "discretization: two DofHandlers on one grid" begin
        for n in (4, 7)
            grid = generate_grid(Triangle, (n, n))
            disc = FerriteDiscretization(grid)   # defaults: P1 for u, P0 for σ
            @test disc isa AbstractDiscretization
            @test ndofs_u(disc) == getnnodes(grid)
            @test ndofs_σ(disc) == getncells(grid)
            @test length(disc.boundary_dofs) == 4n            # nodes on the square boundary
            @test length(disc.boundary_facets) == 4n
            # dof numbering differs from node numbering: check through coordinates
            xb = [coeffs_u(disc, x -> max(abs(x[1]), abs(x[2])))[i] for i in disc.boundary_dofs]
            @test all(≈(1.0), xb)
        end
        grid = generate_grid(Quadrilateral, (3, 3))
        disc = FerriteDiscretization(grid; ip_σ = Lagrange{RefQuadrilateral, 1}())
        @test ndofs_σ(disc) == getnnodes(grid)
        # an explicit boundary (only the left side)
        disc_left = FerriteDiscretization(grid; boundary = getfacetset(grid, "left"))
        @test length(disc_left.boundary_dofs) == 4
    end

    @testset "mass, stiffness and boundary mass matrices ($name)" for (name, grid, ipu, ipσ) in fem_cases()
        disc = FerriteDiscretization(grid; ip_u = ipu, ip_σ = ipσ)
        mats = FEMatrices(disc)
        nu, nσ = ndofs_u(disc), ndofs_σ(disc)
        @test size(mats.M_u) == size(mats.K_u) == size(mats.M_Γ) == (nu, nu)
        @test size(mats.M_σ) == (nσ, nσ)
        @test ones(nu)' * mats.M_u * ones(nu) ≈ SQUARE_AREA
        @test ones(nσ)' * mats.M_σ * ones(nσ) ≈ SQUARE_AREA
        @test ones(nu)' * mats.M_Γ * ones(nu) ≈ SQUARE_PERIMETER
        @test norm(mats.K_u * ones(nu)) < 1e-12
        x1 = coeffs_u(disc, x -> x[1])
        @test x1' * mats.K_u * x1 ≈ SQUARE_AREA                 # ∫ |∇x₁|² = area
        @test issymmetric(mats.M_u) && issymmetric(mats.K_u) && issymmetric(mats.M_σ)
        # the factorised σ mass matrix solves with M_σ
        b = randn(rng, nσ)
        @test mats.M_σ * (mats.M_σ_fac \ b) ≈ b
    end

    @testset "conductivity tensor: nzval(L(σ)) = T σ ($name)" for (name, grid, ipu, ipσ) in fem_cases()
        disc = FerriteDiscretization(grid; ip_u = ipu, ip_σ = ipσ)
        ct = ConductivityTensor(disc)
        nu, nσ = ndofs_u(disc), ndofs_σ(disc)
        @test size(ct.T) == (nnz(ct.pattern), nσ)
        σ = 0.5 .+ rand(rng, nσ)
        L_direct = assemble_weighted_stiffness(disc, σ)       # classical element loop
        L = copy(ct.pattern)
        assemble_weighted_stiffness!(L, ct, σ)                 # one SpMV
        @test L ≈ L_direct
        @test issymmetric(L)
        # σ ≡ 1 reproduces the stiffness matrix
        assemble_weighted_stiffness!(L, ct, ones(nσ))
        @test L ≈ FEMatrices(disc).K_u
        # quadrature is exact: σ = 1 + x₁ (in the σ space), u = x₁  →  ∫σ|∇u|² = ∫(1 + x₁) = 4
        σlin = interpolate_function(disc, x -> 1 + x[1]; field = :σ)
        if ipσ isa DiscontinuousLagrange   # P0/Q0 holds cell means of 1 + x₁: integral still exact
            σlin = l2_project(disc, x -> 1 + x[1]; field = :σ)
        end
        assemble_weighted_stiffness!(L, ct, σlin)
        x1 = coeffs_u(disc, x -> x[1])
        @test x1' * L * x1 ≈ SQUARE_AREA
    end

    @testset "gradient contraction: gₐ = Σₛ λₛᵀ (∂L/∂σₐ) uₛ ($name)" for (name, grid, ipu, ipσ) in fem_cases()
        disc = FerriteDiscretization(grid; ip_u = ipu, ip_σ = ipσ)
        ct = ConductivityTensor(disc)
        nu, nσ, s = ndofs_u(disc), ndofs_σ(disc), 3
        Λ, U = randn(rng, nu, s), randn(rng, nu, s)
        g = zeros(nσ)
        tensor_gradient!(g, ct, Λ, U)
        # L is linear in σ, so gᵀδ = Σₛ λₛᵀ L(δ) uₛ exactly for every direction δ
        δ = randn(rng, nσ)
        Lδ = copy(ct.pattern)
        assemble_weighted_stiffness!(Lδ, ct, δ)
        @test dot(g, δ) ≈ sum(dot(Λ[:, k], Lδ * U[:, k]) for k in 1:s)
        # single columns add up; α, β scaling (g ← α g_new + β g)
        g1 = zeros(nσ)
        for k in 1:s
            tensor_gradient!(g1, ct, Λ[:, k], U[:, k]; β = 1)
        end
        @test g1 ≈ g
        g2 = copy(g)
        tensor_gradient!(g2, ct, Λ, U; α = -2, β = 1)
        @test g2 ≈ -g
    end

    @testset "Riesz maps (discretize-then-optimize vs L² gradient)" begin
        disc = FerriteDiscretization(generate_grid(Triangle, (6, 6)))
        mats = FEMatrices(disc)
        g = randn(rng, ndofs_σ(disc))
        @test riesz_map(CoefficientGradient(), g) == g
        gl2 = riesz_map(L2Gradient(mats), g)
        @test mats.M_σ * gl2 ≈ g
        # P0: M_σ is diagonal with the cell areas
        @test gl2 ≈ g ./ diag(mats.M_σ)
    end

    @testset "coefficient assembly: interpolation and L² projection" begin
        for (name, grid, ipu, ipσ) in fem_cases()
            disc = FerriteDiscretization(grid; ip_u = ipu, ip_σ = ipσ)
            mats = FEMatrices(disc)
            f = x -> 1 + 2x[1] - x[2]
            # linear functions lie in every u space: interpolation and projection agree and are exact
            fi = interpolate_function(disc, f; field = :u)
            fp = l2_project(disc, f; field = :u)
            @test fi ≈ fp
            @test ones(ndofs_u(disc))' * mats.M_u * fi ≈ 4.0      # ∫ f = 4
            # σ space: projection preserves the integral (also for P0 = cell means)
            fσ = l2_project(disc, f; field = :σ)
            @test ones(ndofs_σ(disc))' * mats.M_σ * fσ ≈ 4.0
        end
    end

    @testset "FE-space functionals: inner products, norms, TV" begin
        grid = generate_grid(Quadrilateral, (8, 8))
        disc = FerriteDiscretization(grid)
        dσ = FerriteDiscretization(grid; ip_σ = Lagrange{RefQuadrilateral, 1}())
        one_σ = ones(ndofs_σ(disc))
        @test fe_inner(disc, one_σ, one_σ; field = :σ) ≈ 4.0
        @test fe_norm(disc, one_σ; field = :σ) ≈ 2.0
        x1 = interpolate_function(dσ, x -> x[1]; field = :σ)
        @test fe_norm(dσ, x1; field = :σ, kind = :H1semi)^2 ≈ 4.0
        @test fe_norm(dσ, x1; field = :σ, kind = :H1)^2 ≈ 4.0 + 4 / 3    # ∫|∇x₁|² + ∫x₁²

        # TV of an indicator: total jump along the interface x₁ = 0 (height 2)
        step = interpolate_function(disc, x -> x[1] > 0 ? 1.0 : 0.0; field = :σ)   # P0: exact
        @test total_variation(disc, step) ≈ 2.0
        @test total_variation(disc, 3 .* step) ≈ 6.0
        @test total_variation(disc, one_σ) ≈ 0.0 atol = 1e-12
        # continuous σ: TV = ∫ |∇σ|, for σ = x₁ that is the area
        @test total_variation(dσ, x1) ≈ 4.0
        # smoothed TV: gradient matches finite differences (P0 jumps and continuous σ)
        for (d, σ) in ((disc, rand(rng, ndofs_σ(disc))), (dσ, rand(rng, ndofs_σ(dσ))))
            ε = 1e-2
            g = similar(σ)
            tv = total_variation!(g, d, σ; ε)
            @test tv ≈ total_variation(d, σ; ε)
            δ = randn(rng, length(σ))
            h = 1e-6
            fd = (total_variation(d, σ .+ h .* δ; ε) - total_variation(d, σ .- h .* δ; ε)) / 2h
            @test dot(g, δ) ≈ fd rtol = 1e-6
        end
    end
end
