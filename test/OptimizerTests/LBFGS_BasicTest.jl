using ModularEIT
using Test
using LinearAlgebra

@testset "LBFGS_B test" begin


    # 1. Define the Objective Function and its Gradient
    # f(x) = sum( (x_i - target_i)^2 )
    function test_f(x, target)
        return sum((x .- target) .^ 2)
    end

    # The gradient ∇f = 2 * (x - target)
    function test_g(x, target)
        return 2.0 .* (x .- target)
    end

    # 2. Run the Test
    function run_lbfgs_test()
        # Scenario: Target is at -5.0, but bounds are [0, 1]
        # The optimal solution should be exactly [0, 0, 0]
        n = 3
        x0 = [0.5, 0.5, 0.5]
        target = [-5.0, -5.0, -5.0]

        println("--- Starting Test: Target = -5.0, Bounds = [0, 1] ---")

        # Wrap functions to match the lbfgs_b signature
        f(x) = test_f(x, target)
        g(x) = test_g(x, target)

        result = lbfgs_b(f, g, x0; lb=0.0, ub=1.0, m=5, maxiter=50)

        println("\nFinal x: $result")
        println("Target was -5.0, expected [0.0, 0.0, 0.0] due to lb=0.0")

        if all(isapprox.(result, 0.0, atol=1e-5))
            println("✅ TEST PASSED: Constraints respected.")
            return true
        else
            println("❌ TEST FAILED: Constraints violated or optimization failed.")
            return false
        end
    end

    @test run_lbfgs_test()

end


#=
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
=#
