using ModularEIT
using Test

@testset "LBFGS_BasicQuadratic" begin
    println("Testing LBFGS with simple quadratic function")

    # Simple quadratic: f(x) = ||x - x_target||^2
    x_target = [1.0, 2.0, 3.0]
    f(x) = sum((x - x_target) .^ 2)
    ∇f(x) = 2.0 .* (x - x_target)

    # Start far from solution
    x0 = zeros(3)

    # Run LBFGS
    solution = lbfgs(f, ∇f, x0; m=5, tol=1e-6, maxiter=100)

    # Check solution
    error = norm(solution - x_target)
    println("Final error: $error")
    @test error < 1e-3

    # Check that it's better than starting point
    starting_error = norm(x0 - x_target)
    final_error = norm(solution - x_target)
    println("Starting error: $starting_error")
    println("Final error: $final_error")
    @test final_error < starting_error / 100  # Should improve by at least 100x
end

@testset "LBFGS_NonConvex" begin
    println("Testing LBFGS with non-convex Rosenbrock function")

    # Rosenbrock function: f(x) = (1-x1)^2 + 100(x2-x1^2)^2
    f(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
    ∇f(x) = [
        -2 * (1 - x[1]) - 400 * x[1] * (x[2] - x[1]^2);
        200 * (x[2] - x[1]^2)
    ]

    x0 = [0.0, 0.0]
    solution = lbfgs(f, ∇f, x0; m=10, tol=1e-6, maxiter=500)

    # Check that function value decreased
    f_initial = f(x0)
    f_final = f(solution)
    println("Initial f: $f_initial, Final f: $f_final")
    @test f_final < f_initial
end
