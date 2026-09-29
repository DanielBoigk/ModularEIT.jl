# Jacobian SVD (parameter modes ordered by how well the data determine them) and Gauss–Newton with
# truncated-SVD steps.
using ModularEIT
using Ferrite
using LinearAlgebra
using Random
using Test

# linear least squares r(x) = A x - b
struct LinearLS <: AbstractObjective
    A::Matrix{Float64}
    b::Vector{Float64}
end
ModularEIT.objective_value(q::LinearLS, x::AbstractVector) = sum(abs2, q.A * x - q.b) / 2
function ModularEIT.value_and_gradient!(g::AbstractVector, q::LinearLS, x::AbstractVector)
    r = q.A * x - q.b
    g .= q.A' * r
    return sum(abs2, r) / 2
end
ModularEIT.n_residual(q::LinearLS) = length(q.b)
ModularEIT.residual!(r::AbstractVector, q::LinearLS, x::AbstractVector) = (r .= q.A * x .- q.b)
function ModularEIT.residual_and_jacobian!(r::AbstractVector, J::AbstractMatrix, q::LinearLS, x::AbstractVector)
    r .= q.A * x .- q.b
    copyto!(J, q.A)
    return r, J
end

@testset "truncated SVD" begin
    rng = MersenneTwister(41)
    # ill-conditioned linear problem: singular values 10^0 … 10^-7
    m, n = 30, 20
    Q1 = Matrix(qr(randn(rng, m, m)).Q)
    Q2 = Matrix(qr(randn(rng, n, n)).Q)
    svals = 10.0 .^ range(0, -7; length = n)
    A = Q1[:, 1:n] * Diagonal(svals) * Q2'
    b = randn(rng, m)
    q = LinearLS(A, b)

    @testset "Jacobian SVD in a metric" begin
        w = 0.5 .+ rand(rng, n)
        js = jacobian_svd(q, zeros(n); weights = w)
        @test issorted(js.s; rev = true)
        @test js.s ≈ svd(A ./ sqrt.(w')).S
        @test js.V' * Diagonal(w) * js.V ≈ I                     # modes orthonormal in the metric
        @test js.U * Diagonal(js.s) * js.V' * Diagonal(w) ≈ A      # J = U S Vᵀ W
        B = jacobian_basis(q, zeros(n), 5; weights = w)
        @test B ≈ js.V[:, 1:5]
    end

    @testset "one step = TSVD solution (linear problem)" begin
        for k in (3, 8, n)
            res = minimize(q, zeros(n), TruncatedGaussNewton(; rank = k); maxiter = 1)
            F = svd(A)
            xk = F.V[:, 1:k] * ((F.U[:, 1:k]' * b) ./ F.S[1:k])
            @test res.σ ≈ xk rtol = 1e-8
        end
        # relative threshold: modes with s ≥ rtol s₁
        res = minimize(q, zeros(n), TruncatedGaussNewton(; rtol = 1e-3); maxiter = 1)
        k = count(svals .>= 1e-3)
        F = svd(A)
        @test res.σ ≈ F.V[:, 1:k] * ((F.U[:, 1:k]' * b) ./ F.S[1:k]) rtol = 1e-8
        @test_throws ArgumentError TruncatedGaussNewton(; rank = 0)
        @test_throws ArgumentError TruncatedGaussNewton(; rtol = 2.0)
    end

    @testset "sensitivity scaling" begin
        # damping D = diag(‖A eⱼ‖): between the identity and Marquardt's diag(AᵀA)
        d = vec(sqrt.(sum(abs2, A; dims = 1)))
        λ = 1e-2
        res = minimize(q, zeros(n), GaussNewton(; damping = :linesearch, λ, scaling = :sensitivity); maxiter = 1)
        λabs = λ * maximum(d .^ 2) / maximum(d)          # relative to the largest diagonal entries
        @test res.σ ≈ (A' * A + λabs * Diagonal(d)) \ (A' * b) rtol = 1e-8
        # truncated steps in the sensitivity metric
        r1 = minimize(q, zeros(n), TruncatedGaussNewton(; rank = 6, weights = :sensitivity); maxiter = 1)
        r2 = minimize(q, zeros(n), TruncatedGaussNewton(; rank = 6, weights = d); maxiter = 1)
        @test r1.σ ≈ r2.σ
        @test_throws ArgumentError GaussNewton(; scaling = :foo)
        @test_throws ArgumentError TruncatedGaussNewton(; rank = 2, weights = :foo)
    end

    @testset "EIT: pixels, bounds, discrepancy stop" begin
        disc = FerriteDiscretization(generate_grid(Quadrilateral, (16, 16)))
        fm = ForwardModel(disc, CompleteElectrodeModel(angular_electrodes(disc, 16; coverage = 0.5), 0.1))
        I_ = trigonometric_patterns(fm, 5)
        σtrue = conductivity(disc, InclusionPhantom(1.0, [CircleInclusion((0.3, 0.2), 0.4, 2.5)]))
        clean = forward_neumann(fm, σtrue, I_)[1]
        noise = RelativeGaussianNoise(0.005)
        V = add_noise(clean, noise; rng)
        pp = PixelParametrization(disc, 16, 16)
        data = AdjointStateObjective(fm, I_, V)
        obj = ParametrizedObjective(data, pp)
        θ0 = ones(256)
        js = jacobian_svd(obj, θ0)
        @test length(js.s) == min(n_residual(obj), 256)
        # modes are ordered by depth: the leading ones lie near the boundary (mean distance to the
        # boundary of the mode's energy; the modes beyond ~100 span the numerical null space)
        dist = [1 - max(abs(c[1]), abs(c[2])) for c in pp.centres]
        depth(r) = sum(sum(abs2.(js.V[:, i]) .* dist) for i in r) / length(r)
        @test 2 * depth(1:10) < depth(61:100)
        res = minimize(obj, θ0, TruncatedGaussNewton(; rtol = 1e-2); lower = 0.1, maxiter = 30,
                       ftarget = discrepancy_target(data, noise))
        @test res.value < 1e-2 * objective_value(obj, θ0)
        @test all(res.σ .>= 0.1)
        @test norm(conductivity(pp, res.σ) - σtrue) < norm(ones(ndofs_σ(disc)) - σtrue)
        # penalties and truncation are alternatives
        @test_throws ArgumentError minimize(RegularizedObjective(obj, 1e-3 => TikhonovRegularizer(pp.pixel_disc)),
                                            θ0, TruncatedGaussNewton(; rank = 5))
    end
end
