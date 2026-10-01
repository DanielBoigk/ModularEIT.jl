# Regularizers and regularized objectives: values, gradients (central finite differences) and
# the Gauss–Newton Hessians used by the second-order methods.
using ModularEIT
using ModularEITFerrite
using Ferrite
using SparseArrays
using LinearAlgebra
using Random
using Test

central_fd_reg(f, σ, δ, h) = (f(σ .+ h .* δ) .- f(σ .- h .* δ)) ./ 2h

@testset "regularizers" begin
    rng = MersenneTwister(4)
    tri = generate_grid(Triangle, (6, 6))
    quad = generate_grid(Quadrilateral, (5, 5))
    p0 = FerriteDiscretization(tri)
    p1 = FerriteDiscretization(tri; ip_σ = Lagrange{RefTriangle, 1}())
    q0 = FerriteDiscretization(quad)

    function check_gradient(reg, n; h = 1e-6, rtol = 1e-6)
        σ = 1 .+ rand(rng, n)
        g = zeros(n)
        R = value_and_gradient!(g, reg, σ)
        @test R ≈ objective_value(reg, σ)
        δ = randn(rng, n)
        @test dot(g, δ) ≈ central_fd_reg(s -> objective_value(reg, s), σ, δ, h) rtol = rtol
        return σ, g
    end

    @testset "Tikhonov $kind on $name" for (name, disc) in (("P0", p0), ("P1", p1), ("Q0", q0)),
                                            kind in (:L2, :H1semi, :H1, :jump)
        n = ndofs_σ(disc)
        if kind in (:H1semi, :H1) && name != "P1"
            # K_σ = 0 for piecewise constants: the seminorm would silently vanish
            @test_throws ArgumentError TikhonovRegularizer(disc; kind)
            continue
        end
        if kind === :jump && name == "P1"
            @test_throws ArgumentError TikhonovRegularizer(disc; kind)
            continue
        end
        σ0 = fill(2.0, n)
        reg = TikhonovRegularizer(disc; kind, reference = σ0)
        @test reg isa AbstractRegularizer
        @test objective_value(reg, σ0) ≈ 0 atol = 1e-14
        σ, g = check_gradient(reg, n)
        H = gauss_newton_hessian(reg, σ)
        @test issymmetric(H)
        @test H * (σ - σ0) ≈ g                         # quadratic: exact Hessian
        if kind in (:H1semi, :jump)
            @test objective_value(reg, σ0 .+ 3) ≈ 0 atol = 1e-12   # constants are free
        end
    end

    @testset "Tikhonov: L² value is ½‖σ - σ₀‖²_{L²}" begin
        reg = TikhonovRegularizer(p0; kind = :L2)
        # piecewise constant 1 on the unit square [-1, 1]²: ½ × 4
        @test objective_value(reg, ones(ndofs_σ(p0))) ≈ 2
    end

    @testset "jump penalty = two-point-flux ½|σ|²_{H¹} of a linear function" begin
        disc = FerriteDiscretization(generate_grid(Quadrilateral, (20, 20)))
        h = 0.1
        σ = interpolate_function(disc, x -> 3x[1])     # |∇σ|² = 9 on [-1, 1]²
        reg = TikhonovRegularizer(disc; kind = :jump)
        # Σ_F |F|/d (jump)² integrates 9 over the region between the outermost cell centroids
        @test objective_value(reg, σ) ≈ 0.5 * 9 * (2 - h) * 2 rtol = 1e-10
    end

    @testset "smoothed total variation on $name" for (name, disc) in (("P0", p0), ("P1", p1), ("Q0", q0))
        n = ndofs_σ(disc)
        reg = TotalVariationRegularizer(disc; ε = 1e-2)
        @test reg isa AbstractRegularizer
        σ = 1 .+ rand(rng, n)
        @test objective_value(reg, σ) ≈ total_variation(disc, σ; ε = 1e-2)
        σ, g = check_gradient(reg, n)
        # lagged diffusivity: the Hessian model reproduces the gradient, ∇TV(σ) = H(σ) σ
        H = gauss_newton_hessian(reg, σ)
        @test issymmetric(H)
        @test H * σ ≈ g
        @test minimum(eigvals(Symmetric(Matrix(H)))) > -1e-10
        @test norm(H * ones(n)) < 1e-10                 # constants are free
    end

    @testset "regularized objective" begin
        disc = p0
        n = ndofs_σ(disc)
        fm = ForwardModel(disc, ContinuumModel())
        I_ = trigonometric_patterns(fm, 2)
        V = forward_neumann(fm, 1 .+ rand(rng, n), I_)[1]
        data = AdjointStateObjective(fm, I_, V)
        tik = TikhonovRegularizer(disc; kind = :jump)
        tv = TotalVariationRegularizer(disc; ε = 1e-2)
        obj = RegularizedObjective(data, 1e-3 => tik, 2e-3 => tv)
        @test obj isa AbstractObjective
        σ = 1 .+ rand(rng, n)
        @test objective_value(obj, σ) ≈
              objective_value(data, σ) + 1e-3 * objective_value(tik, σ) + 2e-3 * objective_value(tv, σ)
        g = zeros(n)
        J = value_and_gradient!(g, obj, σ)
        @test J ≈ objective_value(obj, σ)
        δ = randn(rng, n)
        @test dot(g, δ) ≈ central_fd_reg(s -> objective_value(obj, s), σ, δ, 1e-6) rtol = 1e-6

        # data terms must deliver coefficient gradients; the optimisers apply the Riesz map
        data_l2 = AdjointStateObjective(fm, I_, V; gradient = L2Gradient(FEMatrices(disc)))
        @test_throws ArgumentError RegularizedObjective(data_l2, 1.0 => tik)
    end
end
