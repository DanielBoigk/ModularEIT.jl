# Parametrizations σ = P θ: pixel grids, subspaces (DCT modes, boundary bands) and objectives in
# the parameters, on pixel-aligned, triangular, polar and adaptively refined meshes.
using ModularEIT
using ModularEITFerrite
using Ferrite
using SparseArrays
using LinearAlgebra
using Random
using Test

central_fd_par(f, θ, δ, h) = (f(θ .+ h .* δ) .- f(θ .- h .* δ)) ./ 2h

@testset "parametrizations" begin
    rng = MersenneTwister(31)

    @testset "pixels on a pixel-aligned mesh: identity" begin
        disc = FerriteDiscretization(generate_grid(Quadrilateral, (12, 8), Vec(0.0, 0.0), Vec(3.0, 2.0)))
        pp = PixelParametrization(disc, 8, 12)
        @test pp isa AbstractParametrization
        @test parameter_count(pp) == 96
        @test pp.bbox == (0.0, 3.0, 0.0, 2.0)
        θ = rand(rng, 96)
        σ = conductivity(pp, θ)
        @test to_image(disc, σ, 8, 12) ≈ pixel_image(pp, θ)       # same pixel values
        @test all(pp.active)
        @test pixel_parameters(pp, pixel_image(pp, θ)) ≈ θ         # image ↔ parameters
    end

    @testset "constants and bounds are preserved on $name" for (name, disc) in (
            ("finer quads", FerriteDiscretization(generate_grid(Quadrilateral, (24, 24)))),
            ("triangles", FerriteDiscretization(generate_grid(Triangle, (10, 10)))),
            ("P1 triangles", FerriteDiscretization(generate_grid(Triangle, (10, 10)); ip_σ = Lagrange{RefTriangle, 1}())),
            ("polar disk", FerriteDiscretization(polar_grid(8, 64))),
            ("P1 polar disk", FerriteDiscretization(polar_grid(8, 64); ip_σ = Lagrange{RefTriangle, 1}())))
        pp = PixelParametrization(disc, 16, 16)
        P = pp.P
        @test size(P) == (ndofs_σ(disc), 256)
        @test all(>=(0), nonzeros(P))
        @test P * ones(256) ≈ ones(ndofs_σ(disc))                  # partition of unity
        θ = 0.2 .+ 0.7 .* rand(rng, 256)
        σ = conductivity(pp, θ)
        @test all(0.2 - 1e-12 .<= σ .<= 0.9 + 1e-12)                # convex combinations
        # smooth fields: σ ≈ interpolant of the pixel function (first order in the pixel size)
        f(x) = 1 + 0.3x[1] - 0.2x[2]^2
        θf = [f(c) for c in pp.centres]
        @test maximum(abs, conductivity(pp, θf) .- conductivity(disc, f; method = :interpolate)) < 0.15
        # pixels whose centre is far outside the domain do not influence σ
        name in ("polar disk", "P1 polar disk") && @test !all(pp.active)
    end

    @testset "exact pixel values on a mesh finer than the pixels" begin
        disc = FerriteDiscretization(generate_grid(Quadrilateral, (32, 32)))
        pp = PixelParametrization(disc, 8, 8)
        θ = rand(rng, 64)
        @test to_image(disc, conductivity(pp, θ), 8, 8) ≈ pixel_image(pp, θ)
    end

    @testset "adaptively refined mesh (hanging nodes)" begin
        am = AdaptiveMesh(generate_grid(Quadrilateral, (8, 8)))
        refine_mesh!(am, [1, 2, 3, 8, 9, 57, 64])
        disc = FerriteDiscretization(current_grid(am))
        pp = PixelParametrization(disc, 16, 16)
        @test pp.P * ones(256) ≈ ones(ndofs_σ(disc))
    end

    @testset "DCT and boundary-band subspaces" begin
        disc = FerriteDiscretization(polar_grid(8, 64); ip_σ = Lagrange{RefTriangle, 1}())
        pp = PixelParametrization(disc, 16, 16)
        C = dct_basis(pp, 4)
        @test size(C) == (256, 16)
        @test C' * C ≈ I                                            # orthonormal modes
        @test C[:, 1] ≈ fill(1 / 16, 256)                           # the constant mode first
        sp = SubspaceParametrization(pp, C)
        @test parameter_count(sp) == 16
        θ = randn(rng, 16)
        @test conductivity(sp, θ) ≈ conductivity(pp, C * θ)
        @test isapprox(pixel_image(sp, θ), pixel_image(pp, C * θ); nans = true)
        # a constant conductivity is represented exactly
        @test conductivity(sp, [16.0; zeros(15)]) ≈ ones(ndofs_σ(disc))
        # boundary band: one indicator per active pixel within `width` of the boundary
        Bb = boundary_band_basis(pp, 0.2)
        @test all(sum(Bb; dims = 1) .== 1)
        @test 0 < size(Bb, 2) < count(pp.active)
        dists = [minimum(norm(c - Vec(cos(t), sin(t))) for t in range(0, 2π; length = 400)) for c in pp.centres]
        band = vec(sum(Bb; dims = 2)) .> 0
        @test all(dists[band] .<= 0.2 + 0.05)                       # (boundary is a 64-gon)
        # hybrid: low DCT modes everywhere plus free pixels in the band
        hy = SubspaceParametrization(pp, [C Bb])
        @test parameter_count(hy) == 16 + size(Bb, 2)
    end

    @testset "objective in the parameters: gradient, Jacobian, reconstruction" begin
        disc = FerriteDiscretization(generate_grid(Quadrilateral, (16, 16)))
        fm = ForwardModel(disc, CompleteElectrodeModel(angular_electrodes(disc, 8; coverage = 0.5), 0.1))
        I_ = trigonometric_patterns(fm, 3)
        σtrue = conductivity(disc, InclusionPhantom(1.0, [CircleInclusion((0.3, 0.2), 0.35, 2.5)]))
        V = forward_neumann(fm, σtrue, I_)[1]
        pp = PixelParametrization(disc, 8, 8)
        for par in (pp, SubspaceParametrization(pp, dct_basis(pp, 3)))
            obj = ParametrizedObjective(AdjointStateObjective(fm, I_, V), par)
            @test obj isa AbstractObjective
            p = parameter_count(par)
            θ = par === pp ? 1 .+ 0.3 .* rand(rng, p) : [8.0; 0.3 .* randn(rng, p - 1)]
            g = zeros(p)
            J = value_and_gradient!(g, obj, θ)
            @test J ≈ objective_value(AdjointStateObjective(fm, I_, V), conductivity(par, θ))
            δ = randn(rng, p)
            @test dot(g, δ) ≈ central_fd_par(t -> objective_value(obj, t), θ, δ, 1e-6) rtol = 1e-6
            m = n_residual(obj)
            r, Jm = zeros(m), zeros(m, p)
            residual_and_jacobian!(r, Jm, obj, θ)
            @test Jm * δ ≈ central_fd_par(t -> residual!(zeros(m), obj, t), θ, δ, 1e-6) rtol = 1e-6
            @test Jm' * r ≈ g rtol = 1e-8
        end
        # regularizers act on the pixel discretization; Gauss–Newton in pixel space
        obj = RegularizedObjective(ParametrizedObjective(AdjointStateObjective(fm, I_, V), pp),
                                   1e-5 => TotalVariationRegularizer(pp.pixel_disc; ε = 1e-2))
        θ0 = ones(64)
        res = minimize(obj, θ0, GaussNewton(); lower = 0.1, maxiter = 20)
        @test res.value < 1e-2 * objective_value(obj, θ0)
        @test all(conductivity(pp, res.σ) .>= 0.1 - 1e-12)

        # subspace parameters cannot be bounded: trial steps with negative conductivity are
        # rejected (InfeasibleConductivityError) instead of failing
        @test_throws InfeasibleConductivityError system_matrix!(fm, -ones(ndofs_σ(disc)))
        hy = SubspaceParametrization(pp, [dct_basis(pp, 3) boundary_band_basis(pp, 0.3)])
        objh = ParametrizedObjective(AdjointStateObjective(fm, I_, V), hy)
        θh = [8.0; zeros(parameter_count(hy) - 1)]
        rh = minimize(objh, θh, GaussNewton(; λ = 1e-8); maxiter = 15)
        @test rh.value < objective_value(objh, θh)
        @test all(conductivity(hy, rh.σ) .> 0)
    end
end
