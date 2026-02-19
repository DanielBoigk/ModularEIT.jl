# Gradient Testing Guide for ModularEIT

## Overview

Gradient testing is a crucial validation technique for checking the correctness of gradient computations in optimization algorithms. This guide explains how to perform gradient tests for the Neumann adjoint solver in ModularEIT.

## Mathematical Background

### The Optimization Problem

In EIT inverse problems, we minimize an objective function:

$$J(\sigma) = d(b(\sigma), f)$$

where:
- **σ** is the conductivity field (the optimization variable)
- **u(σ)** solves the forward PDE: $\nabla \cdot (\sigma \nabla u) = 0$ with Neumann BC $\sigma \frac{\partial u}{\partial n} = g$
- **b(σ)** is the boundary trace of the solution: $b(\sigma) = u(\sigma)|_{\partial \Omega}$
- **f** is the target boundary data (measured voltages)
- **d(·,·)** is the data misfit (typically $d(u,v) = \|u - v\|^2$)

### The Gradient

The gradient of the objective with respect to conductivity is:

$$\frac{\partial J}{\partial \sigma} = -\nabla u \cdot \nabla \lambda$$

where **λ** is the adjoint solution to:

$$\nabla \cdot (\sigma \nabla \lambda) = 0 \quad \text{with} \quad \sigma \frac{\partial \lambda}{\partial n} = \frac{\partial d}{\partial u}(b, f)$$

## Gradient Test Theory

### Directional Derivative

For any search direction $\delta\sigma$, the directional derivative is:

$$J'(\sigma; \delta\sigma) = \left\langle \nabla J(\sigma), \delta\sigma \right\rangle = \int_{\Omega} \frac{\partial J}{\partial \sigma} \cdot \delta\sigma \, dx$$

### Finite Difference Approximation

For small $\epsilon > 0$:

$$J'(\sigma; \delta\sigma) \approx \frac{J(\sigma + \epsilon \delta\sigma) - J(\sigma)}{\epsilon}$$

### Convergence Rate

If the gradient is **correctly computed**, the finite difference error should decay as:

$$\left| J'(\sigma; \delta\sigma) - \frac{J(\sigma + \epsilon \delta\sigma) - J(\sigma)}{\epsilon} \right| = O(\epsilon)$$

This means the convergence rate (when plotting error vs. $\epsilon$ on a log-log scale) should be approximately **1.0**.

## Running the Gradient Test

### Quick Start

Run the gradient test for the Neumann adjoint solver:

```julia
using ModularEIT

include("test/GradientTests/NeumannAdjointGradientTest.jl")
```

Or through the test suite:

```bash
cd /path/to/ModularEIT
julia --project -e "using Pkg; Pkg.test(\"ModularEIT\")"
```

### Understanding the Output

The test will output something like:

```
======================================================================
Neumann Adjoint Gradient Test
======================================================================

1. Setting up FEM space and modes...
   - Grid: 15×15
   - FE space dimension: 225
   - Boundary DOFs: 60
   - Number of modes: 3

2. Setting up solver state...

3. Computing objective and gradient at test point...
   - Objective value: 12.345
   - Gradient norm: 6.789

4. Running finite-difference gradient test...

   Testing direction 1 of 2...
      Gradient norm: 6.789
      Convergence rate: 1.05 (expect ≈ 1.0)
        ε = 0.01: error = 0.0234
        ε = 0.001: error = 0.00234

   Testing direction 2 of 2...
      ...

======================================================================
GRADIENT TEST SUMMARY
======================================================================
Average convergence rate: 1.03
Expected convergence rate: ≈ 1.0 (for first-order accurate gradient)

✓ PASS: Gradient is first-order accurate
```

## Interpreting Results

### Convergence Rate Analysis

Plot the error $e(\epsilon) = |J' - (J(\sigma + \epsilon\delta\sigma) - J(\sigma))/\epsilon|$ vs. $\epsilon$:

- **Rate ≈ 1.0**: ✓ Gradient is correct (first-order accurate)
- **Rate ≈ 0.5**: ⚠ Indicates second-order error; possible issues with gradient
- **Rate ≈ 2.0**: ⚠ Indicates discretization error dominates; try smaller $\epsilon$ or coarser grid
- **Rate < 0.5 or scattered**: ✗ Gradient is likely incorrect

### Common Issues

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| Rate >> 1.0 | Discretization error in FD | Use coarser grid or smaller number of modes |
| Rate << 1.0 | Gradient implementation error | Check sign, assembly, boundary conditions |
| Scattered convergence | Ill-conditioning or rounding errors | Use higher precision, better-scaled problem |
| Negative gradient error | Sign error in gradient (e.g., missing minus sign) | Review adjoint assembly |

## How to Write Your Own Gradient Test

### Template

```julia
using ModularEIT
using LinearAlgebra
using Random

function my_gradient_test()
    # 1. Setup: Create FEM space and problem
    grid = generate_grid(Quadrilateral, (15, 15))
    ∂Ω = union(getfacetset.((grid,), ["left", "top", "right", "bottom"])...)
    fe = FerriteFESpace{RefQuadrilateral}(grid, 2, 3, ∂Ω)
    
    # 2. Create test problem with known properties
    σ_ref = ones(fe.n)
    K_ref = assemble_L(fe, σ_ref)
    K_fac = factorize(K_ref)
    
    # Create a few modes
    modes = Dict{Int, FerriteEITMode}()
    for i in 1:3
        g = randn(fe.m); g .-= mean(g)
        modes[i] = create_mode_from_g(fe, fe.up(g), K_fac)
        modes[i].f = randn(fe.m); modes[i].f .-= mean(modes[i].f)
        modes[i].F = fe.up(modes[i].f)
    end
    
    # 3. Setup solver state
    σ_test = 0.5 .+ 0.1 .* randn(fe.n)
    σ_test = max.(σ_test, 1e-6)
    
    sol = FerriteSolverState(fe, σ_test)
    prblm = FerriteProblem(fe, modes, sol)
    
    # 4. Compute reference objective and gradient
    for (i, mode) in modes
        objective_neumann_init!(mode, sol, fe)
        gradient_neumann_init!(mode, sol, fe)
    end
    
    J_ref = sum(m.error_n for m in values(modes))
    ∇J = sum(m.δσ for m in values(modes))
    
    # 5. Test with finite differences
    δσ = randn(fe.n); δσ .-= mean(δσ); δσ ./= norm(δσ)
    J_prime = dot(∇J, δσ)
    
    epsilons = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    errors = []
    
    for ε in epsilons
        σ_pert = σ_test .+ ε .* δσ
        σ_pert = max.(σ_pert, 1e-6)
        
        sol.σ .= σ_pert
        update_L!(sol, fe, true)
        
        J_pert = 0.0
        for (i, mode) in modes
            objective_neumann_init!(mode, sol, fe)
            J_pert += mode.error_n
        end
        
        J_fd = (J_pert - J_ref) / ε
        push!(errors, abs(J_prime - J_fd))
    end
    
    # 6. Analyze convergence
    rate = polyfit(log.(epsilons), log.(errors), 1)[1]
    println("Convergence rate: $rate (expect ≈ 1.0)")
    
    return 0.8 < rate < 1.2
end
```

## Advanced: Testing Individual Components

### Testing Just the Adjoint Solver

```julia
# Manually construct the adjoint RHS and solve
mode = modes[1]
sol = prblm.state

# Compute adjoint RHS: ∂d/∂u(b, f)
rhs_data = sol.∂d(mode.b, mode.f)
rhs_lifted = fe.up(rhs_data)

# Normalize
mean_boundary!(rhs_lifted, mode, fe.down)

# Solve adjoint: ∇·(σ∇λ) = 0
mode.λ = sol.L_fac \ rhs_lifted
mode.λ .-= mean(mode.λ)

# Check: λ should satisfy boundary condition to machine precision
```

### Testing the Sensitivity Calculation

```julia
# Verify the bilinear form computation
mode.δσ_test = calculate_bilinear_map!(fe, mode.rhs, mode.λ, mode.u_g)

# This should equal ∫_Ω ∇u · ∇λ dx (up to sign and assembly details)
```

## Best Practices

1. **Use multiple random directions**: Test at least 2-3 different search directions to rule out lucky cancellations
2. **Vary the problem size**: Test on grids of different sizes (15×15, 31×31, etc.)
3. **Use multiple modes**: Include several measurement modes to catch errors in mode aggregation
4. **Check edge cases**: Test at points where the solution might have features (e.g., near conductivity boundaries)
5. **Document expected behavior**: Note what convergence rate you expect for your specific problem
6. **Version control**: Include gradient tests in CI/CD pipelines to catch regressions

## References

- Gradient checking is described in detail in Boyd & Vandenberghe, "Convex Optimization" (Appendix A.4)
- Adjoint methods: Plessix, "A review of the adjoint-state method for computing the gradient of a functional with geophysical applications" (2006)
- For EIT specifically: Lionheart & Somersalo, "An asymptotic formula for the electric potential in an anisotropic medium" (2014)

## Troubleshooting

### Test keeps failing with rate ≈ 0.5

**Likely cause**: Discretization error dominates over the Taylor error.

**Solutions**:
- Use a coarser grid (e.g., 13×13 instead of 31×31)
- Reduce the number of modes
- Start with larger epsilon values (1e-1 instead of 1e-7)

### Test passes on coarse grids but fails on fine grids

**Likely cause**: Gradient implementation is correct but has scaling issues.

**Solutions**:
- Check if the gradient should be weighted by cell volumes or Jacobian determinants
- Verify boundary condition handling in the adjoint equation
- Review assembly of the bilinear form

### Convergence rate is ~2.0

**Likely cause**: The finite difference is second-order accurate (rare but possible).

**Solutions**:
- Check if you're using centered differences instead of forward differences
- Verify the gradient formula isn't implementing a better approximation
- Try very small epsilon to see if it improves

### Random failures or very noisy results

**Likely cause**: Ill-conditioned problem or round-off errors.

**Solutions**:
- Scale the problem (multiply σ by a constant, adjust epsilon accordingly)
- Use double precision throughout
- Avoid modes that are nearly singular