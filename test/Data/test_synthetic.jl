# Synthetic conductivities (inclusion phantoms, Gaussian random fields, images), data simulation
# without inverse crime, and spectral image corruption.
using ModularEIT
using ModularEITFerrite
using Ferrite
using LinearAlgebra
using Statistics
using Random
using Test

@testset "synthetic data" begin
    @testset "inclusions" begin
        c = CircleInclusion((0.2, -0.1), 0.3, 2.0)
        @test Vec(0.2, -0.1) in c
        @test !(Vec(0.6, -0.1) in c)
        @test c.value == 2.0
        e = EllipseInclusion((0.0, 0.0), (0.5, 0.1), π / 2, 3.0)          # rotated: long axis along y
        @test Vec(0.0, 0.45) in e
        @test !(Vec(0.45, 0.0) in e)
        p = PolygonInclusion([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], 0.5)
        @test Vec(0.2, 0.2) in p
        @test !(Vec(0.6, 0.6) in p)
        ph = InclusionPhantom(1.0, [c, CircleInclusion((0.2, -0.1), 0.1, 5.0)])  # later inclusions on top
        @test ph(Vec(0.2, -0.1)) == 5.0
        @test ph(Vec(0.2, 0.15)) == 2.0
        @test ph(Vec(-0.9, 0.9)) == 1.0
    end

    @testset "random inclusions: $domain" for domain in (:disk, :box)
        for k in 1:20
            ph = random_inclusions(MersenneTwister(k); domain, count = 1:4, values = (0.2, 5.0), margin = 0.05)
            @test ph isa InclusionPhantom
            @test 1 <= length(ph.inclusions) <= 4
            for inc in ph.inclusions
                @test 0.2 <= inc.value <= 5.0
                # the bounding circle (with margin) lies inside the domain, checked at 64 × 3 points
                pts = [ModularEIT._center(inc) .+ r .* ModularEIT._bounding_radius(inc) .* (cos(θ), sin(θ))
                       for θ in range(0, 2π; length = 64), r in (0.0, 0.5, 1.0)]
                inside(x) = domain === :disk ? norm(x) <= 1 - 0.05 + 1e-12 : all(abs.(x) .<= 1 - 0.05 + 1e-12)
                @test all(inside, pts)
            end
            # no overlaps (bounding circles)
            incs = ph.inclusions
            for i in eachindex(incs), j in 1:(i - 1)
                d = norm(ModularEIT._center(incs[i]) .- ModularEIT._center(incs[j]))
                @test d >= ModularEIT._bounding_radius(incs[i]) + ModularEIT._bounding_radius(incs[j])
            end
        end
        @test random_inclusions(MersenneTwister(3); domain).inclusions ==
              random_inclusions(MersenneTwister(3); domain).inclusions
    end

    @testset "projection onto a mesh: cell averages" begin
        disc = FerriteDiscretization(generate_grid(Quadrilateral, (40, 40)))
        mats = FEMatrices(disc)
        ph = InclusionPhantom(1.0, [CircleInclusion((0.1, 0.2), 0.4, 3.0)])
        σ = conductivity(disc, ph)
        @test sum(mats.M_σ * σ) ≈ 4 + (3 - 1) * π * 0.4^2 rtol = 2e-3   # ∫σ
        @test all(isapprox.(extrema(σ), (1.0, 3.0); atol = 1e-12))
        @test conductivity(disc, σ) === σ                         # coefficient vectors pass through
        # nodal interpolation of a phantom (a callable struct) into a continuous σ space
        p1 = FerriteDiscretization(generate_grid(Triangle, (10, 10)); ip_σ = Lagrange{RefTriangle, 1}())
        σi = conductivity(p1, ph; method = :interpolate)
        @test sort(unique(σi)) == [1.0, 3.0]
    end

    @testset "Gaussian random fields" begin
        box = (-1.0, 1.0, -1.0, 1.0)
        f = gaussian_random_field(MersenneTwister(1); bbox = box, ℓ = 0.2)
        @test f isa PixelFunction
        @test f(Vec(0.3, 0.3)) == gaussian_random_field(MersenneTwister(1); bbox = box, ℓ = 0.2)(Vec(0.3, 0.3))
        # statistics over samples: zero mean, unit variance, correlation decays over a few ℓ
        pts = [Vec(x, 0.0) for x in (-0.5, -0.48, 0.5)]           # distances 0.02 = 0.1ℓ and 1.0 = 5ℓ
        vals = [let g = gaussian_random_field(MersenneTwister(10 + k); bbox = box, ℓ = 0.2); g.(pts) end
                for k in 1:300]
        V = reduce(hcat, vals)
        @test abs(mean(V[1, :])) < 0.2
        @test var(V[1, :]) ≈ 1 rtol = 0.25
        @test cor(V[1, :], V[2, :]) > 0.95
        @test abs(cor(V[1, :], V[3, :])) < 0.2
        # rougher fields (smaller order) decorrelate faster at short range
        rough = [let g = gaussian_random_field(MersenneTwister(10 + k); bbox = box, ℓ = 0.2, order = 1.5); g.(pts) end
                 for k in 1:300]
        R = reduce(hcat, rough)
        @test cor(R[1, :], R[2, :]) < cor(V[1, :], V[2, :])
        # conductivities from fields
        ln = lognormal_phantom(f; σ0 = 2.0, s = 0.5)
        @test ln(Vec(0.1, 0.2)) ≈ 2.0 * exp(0.5 * f(Vec(0.1, 0.2)))
        ls = levelset_phantom(f; level = 0.0, inside = 3.0, outside = 1.0)
        @test ls(Vec(0.1, 0.2)) == (f(Vec(0.1, 0.2)) > 0 ? 3.0 : 1.0)
    end

    @testset "images" begin
        img = [0.0 0.5; 1.0 0.25]                                   # 2 × 2, row 1 on top
        ph = image_phantom(img, 1.0, 3.0; bbox = (-1.0, 1.0, -1.0, 1.0))
        @test ph(Vec(-0.5, 0.5)) ≈ 1.0                              # top left
        @test ph(Vec(0.5, 0.5)) ≈ 2.0                               # top right
        @test ph(Vec(-0.5, -0.5)) ≈ 3.0                             # bottom left
        # on a pixel-aligned mesh the cell averages are the pixel values (same as from_image)
        rng = MersenneTwister(2)
        big = rand(rng, 16, 16)
        disc = FerriteDiscretization(generate_grid(Quadrilateral, (16, 16)))
        @test conductivity(disc, image_phantom(big, 0.5, 2.0)) ≈ 0.5 .+ 1.5 .* from_image(disc, big)
        @test_throws ArgumentError image_phantom(big, 2.0, 1.0)
    end

    @testset "simulation on a finer mesh (no inverse crime)" begin
        ph = InclusionPhantom(1.0, [CircleInclusion((0.3, 0.2), 0.35, 2.0)])
        fine = FerriteDiscretization(generate_grid(Quadrilateral, (48, 48)))
        coarse = FerriteDiscretization(generate_grid(Quadrilateral, (16, 16)))
        # the same physical electrodes on both meshes
        els = angular_electrodes(coarse, 16; coverage = 0.5)
        els_fine = transfer_electrodes(coarse, els, fine)
        @test [electrode_length(fine, e) for e in els_fine] ≈ [electrode_length(coarse, e) for e in els]
        fmf = ForwardModel(fine, CompleteElectrodeModel(els_fine, 0.1))
        fmc = ForwardModel(coarse, CompleteElectrodeModel(els, 0.1))
        I_ = trigonometric_patterns(fmc, 4)
        @test trigonometric_patterns(fmf, 4) ≈ I_                  # electrode patterns are mesh independent
        res = simulate_data(fine, fmf, ph, I_)
        @test res.data == res.clean
        @test length(res.σ) == ndofs_σ(fine)
        coarse_data = forward_neumann(fmc, conductivity(coarse, ph), I_)[1]
        rel = norm(res.data - coarse_data) / norm(res.data)
        @test 1e-5 < rel < 0.05                                     # model error: small but not zero
        noisy = simulate_data(fine, fmf, ph, I_; noise = RelativeGaussianNoise(0.01), rng = MersenneTwister(4))
        @test noisy.clean ≈ res.clean
        @test norm(noisy.data - noisy.clean) / norm(noisy.clean) ≈ 0.01 rtol = 0.3
        @test_throws ArgumentError simulate_data(fine, fmf, ph, I_; mode = :robin)
    end

    @testset "spectral image corruption" begin
        rng = MersenneTwister(5)
        img = zeros(32, 32)
        img[9:24, 9:24] .= 1
        @test corrupt_image(img; rng = MersenneTwister(6), steps = 0) == img
        c = corrupt_image(img; rng = MersenneTwister(6), steps = 3, spatial_noise = 0.0, spectral_noise = 0.0,
                          damping = 1e-3, exponent = 1)
        # damping only: the discrete heat semigroup, mean preserved, edges smoothed, maximum principle
        @test mean(c) ≈ mean(img) atol = 1e-12
        @test maximum(c) <= 1 + 1e-12 && minimum(c) >= -1e-12
        @test norm(diff(c; dims = 1)) < norm(diff(img; dims = 1))
        @test corrupt_image(img; rng = MersenneTwister(6), steps = 2) == corrupt_image(img; rng = MersenneTwister(6), steps = 2)
        noisy = corrupt_image(img; rng = MersenneTwister(7), steps = 2, spatial_noise = 0.1, spectral_noise = 0.0, damping = 0.0)
        @test std(noisy - img) ≈ 0.1 * sqrt(2) rtol = 0.1        # two white noise steps, orthonormal DCT
    end
end
