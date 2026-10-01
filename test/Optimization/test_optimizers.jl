# Optimisers: gradient descent, L-BFGS and Gauss–Newton / Levenberg–Marquardt, unconstrained and
# with box constraints, with Euclidean and non-diagonal Riesz maps, on model problems and on EIT.
using ModularEIT
using ModularEITFerrite
using Ferrite
using SparseArrays
using LinearAlgebra
using Random
using Test

# Rosenbrock as a least-squares problem: r = (10(x₂ - x₁²), 1 - x₁), f = ½‖r‖², minimum at (1, 1)
struct Rosenbrock <: AbstractObjective end
_rr(x) = [10 * (x[2] - x[1]^2), 1 - x[1]]
_rj(x) = [-20x[1] 10.0; -1.0 0.0]
ModularEIT.objective_value(::Rosenbrock, x::AbstractVector) = sum(abs2, _rr(x)) / 2
function ModularEIT.value_and_gradient!(g::AbstractVector, ::Rosenbrock, x::AbstractVector)
    r = _rr(x)
    g .= _rj(x)' * r
    return sum(abs2, r) / 2
end
ModularEIT.n_residual(::Rosenbrock) = 2
ModularEIT.residual!(r::AbstractVector, ::Rosenbrock, x::AbstractVector) = copyto!(r, _rr(x))
function ModularEIT.residual_and_jacobian!(r::AbstractVector, J::AbstractMatrix, ::Rosenbrock, x::AbstractVector)
    copyto!(r, _rr(x))
    copyto!(J, _rj(x))
    return r, J
end

# ½ (x - c)ᵀ A (x - c) as a least-squares problem r = U (x - c), UᵀU = A
struct Quadratic <: AbstractObjective
    A::Matrix{Float64}
    U::Matrix{Float64}
    c::Vector{Float64}
end
Quadratic(A, c) = Quadratic(A, cholesky(Symmetric(A)).U, c)
ModularEIT.objective_value(q::Quadratic, x::AbstractVector) = dot(x - q.c, q.A, x - q.c) / 2
function ModularEIT.value_and_gradient!(g::AbstractVector, q::Quadratic, x::AbstractVector)
    g .= q.A * (x - q.c)
    return dot(x - q.c, g) / 2
end
ModularEIT.n_residual(q::Quadratic) = length(q.c)
ModularEIT.residual!(r::AbstractVector, q::Quadratic, x::AbstractVector) = mul!(r, q.U, x - q.c)
function ModularEIT.residual_and_jacobian!(r::AbstractVector, J::AbstractMatrix, q::Quadratic, x::AbstractVector)
    mul!(r, q.U, x - q.c)
    copyto!(J, q.U)
    return r, J
end

methods_all() = (("gradient descent", GradientDescent()), ("L-BFGS", LBFGS()),
                 ("Gauss–Newton (LM)", GaussNewton()), ("Gauss–Newton (line search)", GaussNewton(; damping = :linesearch)))

@testset "optimizers" begin
    rng = MersenneTwister(11)

    @testset "Rosenbrock: $name" for (name, method) in methods_all()
        maxiter = method isa GradientDescent ? 20_000 : 500
        res = minimize(Rosenbrock(), [-1.2, 1.0], method; maxiter, gtol = 1e-10)
        @test res isa OptimizationState
        @test res isa AbstractSolutionState
        @test res.converged
        @test res.σ ≈ [1.0, 1.0] atol = 1e-5
        @test res.value < 1e-10
        # monotone methods
        vals = [h.value for h in res.history]
        @test all(diff(vals) .<= 1e-14)
        @test res.iteration == length(res.history) - 1
    end

    @testset "box constraints: $name" for (name, method) in methods_all()
        n = 12
        B = randn(rng, n, n)
        A = B'B + I
        c = randn(rng, n)
        lo, hi = -0.3, 0.4
        q = Quadratic(A, c)
        res = minimize(q, zeros(n), method; lower = lo, upper = hi, maxiter = 5000, gtol = 1e-10)
        @test all(lo .<= res.σ .<= hi)
        # KKT: projected gradient vanishes
        g = A * (res.σ - c)
        pg = res.σ .- clamp.(res.σ .- g, lo, hi)
        @test norm(pg) < 1e-6
        # bounds may be vectors, and an infeasible start is projected
        res2 = minimize(q, fill(5.0, n), method; lower = fill(lo, n), upper = fill(hi, n), maxiter = 5000, gtol = 1e-10)
        @test res2.σ ≈ res.σ atol = 1e-5
    end

    @testset "non-diagonal Riesz map: $name" for (name, method) in (("gradient descent", GradientDescent),
                                                                    ("L-BFGS", LBFGS))
        n = 10
        M = spdiagm(0 => fill(4.0, n), 1 => ones(n - 1), -1 => ones(n - 1)) ./ 6
        riesz = L2Gradient(cholesky(Symmetric(M)))
        q = Quadratic(Matrix(M) * 3 + 0.1I, randn(rng, n))
        res = minimize(q, zeros(n), method(; riesz); maxiter = 2000, gtol = 1e-10)
        @test res.σ ≈ q.c atol = 1e-6
        res = minimize(q, zeros(n), method(; riesz); lower = -0.2, upper = 0.2, maxiter = 2000, gtol = 1e-10)
        g = q.A * (res.σ - q.c)
        @test norm(res.σ .- clamp.(res.σ .- g, -0.2, 0.2)) < 1e-6
    end

    @testset "stopping criteria and callbacks" begin
        q = Quadratic(diagm([1.0, 10.0, 100.0]), [1.0, 2.0, 3.0])
        res = minimize(q, zeros(3), GradientDescent(); maxiter = 1000, gtol = 0, ftarget = 0.5)
        @test res.value <= 0.5
        @test res.status === :ftarget
        res = minimize(q, zeros(3), LBFGS(); maxiter = 1)
        @test res.status === :maxiter
        @test !res.converged
        seen = Int[]
        res = minimize(q, zeros(3), GradientDescent(); gtol = 0, callback = st -> (push!(seen, st.iteration); st.iteration >= 2))
        @test seen == [0, 1, 2]
        @test res.status === :callback
        # objectives must deliver coefficient gradients (the optimiser owns the Riesz map)
        disc = FerriteDiscretization(generate_grid(Triangle, (4, 4)))
        fm = ForwardModel(disc, ContinuumModel())
        I_ = trigonometric_patterns(fm, 1)
        V = forward_neumann(fm, ones(ndofs_σ(disc)), I_)[1]
        obj = AdjointStateObjective(fm, I_, V; gradient = L2Gradient(FEMatrices(disc)))
        @test_throws ArgumentError minimize(obj, ones(ndofs_σ(disc)), LBFGS())
        # GN needs residuals
        kv = KohnVogeliusObjective(fm, I_, V)
        @test_throws ArgumentError minimize(kv, ones(ndofs_σ(disc)), GaussNewton())
    end

    @testset "EIT reconstruction: $name" for (name, method) in methods_all()
        disc = FerriteDiscretization(generate_grid(Quadrilateral, (16, 16)))
        mats = FEMatrices(disc)
        n = ndofs_σ(disc)
        fm = ForwardModel(disc, CompleteElectrodeModel(angular_electrodes(disc, 16; coverage = 0.5), 0.1))
        σtrue = interpolate_function(disc, x -> 1 + (norm(x - Vec(0.3, 0.2)) < 0.4))
        I_ = trigonometric_patterns(fm, 7)
        V = forward_neumann(fm, σtrue, I_)[1]
        data = AdjointStateObjective(fm, I_, V)
        obj = RegularizedObjective(data, 1e-6 => TikhonovRegularizer(disc; kind = :jump))
        σ0 = ones(n)
        J0 = objective_value(obj, σ0)
        opt = method isa GradientDescent ? GradientDescent(; riesz = L2Gradient(mats)) :
              method isa LBFGS ? LBFGS(; riesz = L2Gradient(mats)) : method
        res = minimize(obj, σ0, opt; lower = 0.1, maxiter = method isa GaussNewton ? 20 : 300)
        @test res.value < 1e-3 * J0
        @test fe_norm(disc, res.σ - σtrue; mats) < 0.8 * fe_norm(disc, σ0 - σtrue; mats)
        @test all(res.σ .>= 0.1)
    end

    @testset "Gauss–Newton linear solvers agree" begin
        disc = FerriteDiscretization(generate_grid(Triangle, (6, 6)))
        n = ndofs_σ(disc)
        fm = ForwardModel(disc, ContinuumModel())
        I_ = trigonometric_patterns(fm, 3)
        V = forward_neumann(fm, 1 .+ rand(rng, n), I_)[1]
        obj = RegularizedObjective(AdjointStateObjective(fm, I_, V), 1e-4 => TikhonovRegularizer(disc; kind = :L2))
        r1 = minimize(obj, ones(n), GaussNewton(; linear_solver = :dense); maxiter = 3)
        r2 = minimize(obj, ones(n), GaussNewton(; linear_solver = :woodbury); maxiter = 3)
        @test r1.σ ≈ r2.σ rtol = 1e-8
    end
end
