# Discretization

```@meta
CurrentModule = ModularEIT
```

The potential ``u`` and the conductivity ``\sigma`` live in separate finite element spaces on
one mesh (two Ferrite `DofHandler`s). Any pair of Ferrite interpolations works: P1/P0 is the
default, P2/P1 on triangles and Q1/Q0 on the quadrilateral meshes of pixel images are tested.
The quadrature rule integrates ``\int \sigma \nabla\varphi_i\cdot\nabla\varphi_j`` exactly on
affine cells.

```@docs
AbstractDiscretization
FerriteDiscretization
ndofs_u
ndofs_σ
```

## Matrices

```@docs
FEMatrices
assemble_mass!
assemble_stiffness!
assemble_boundary_mass!
assemble_boundary_load!
```

## Conductivity tensor

``L(\sigma)`` is linear in ``\sigma``, so its stored values are ``T\sigma`` for one sparse matrix
``T`` (``\mathrm{nnz}(L)\times n_\sigma``) built once per mesh. The same ``T`` gives every
conductivity derivative ``\partial_{\sigma_a}(\lambda^\top L(\sigma)\,u) = (T^\top w)_a`` with
``w_k = \lambda_{\mathrm{row}_k} u_{\mathrm{col}_k}``, i.e. the discrete adjoint-state gradient
``\int \psi_a \nabla u\cdot\nabla\lambda``. Assembly and gradients are sparse matrix-vector
products and one parallel gather, so they run on the CPU and on every GPU backend
(`to_device = device_converter(ArrayType)`). Theory: wiki article [Conductivity Tensor](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Conductivity-Tensor).

```@docs
ConductivityTensor
assemble_weighted_stiffness!
weighted_stiffness_values!
pair_products!
tensor_gradient!
```

## Gradient representation

```@docs
AbstractRieszMap
CoefficientGradient
L2Gradient
riesz_map!
```

## Coefficients, norms and total variation

```@docs
interpolate_function
l2_project
fe_inner
fe_norm
total_variation
```

## Wiki articles

Theory behind this page in the [theory wiki](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/):

- [Total Variation](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/03-Regularization/Total-Variation)
- [Smoothed Total Variation](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/03-Regularization/Smoothed-Total-Variation)
- [Galerkin Method](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Galerkin-Method)
- [Lagrange Finite Elements](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Lagrange-Finite-Elements)
- [Mass Matrix](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Mass-Matrix)
- [Stiffness Matrix](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Stiffness-Matrix)
- [Weighted Stiffness Matrix](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Weighted-Stiffness-Matrix)
- [Conductivity Tensor](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Conductivity-Tensor)
- [Boundary Mass and Stiffness Matrices](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Boundary-Mass-and-Stiffness-Matrices)
- [L2 Projection](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/L2-Projection)
- [Functional Derivative of the Data Misfit](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/05-Adjoint-Gradients/Functional-Derivative-of-the-Data-Misfit)
- [Gradient Representation and the Riesz Map](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/05-Adjoint-Gradients/Gradient-Representation-and-the-Riesz-Map)
- [Discretize-then-Optimize vs Optimize-then-Discretize](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/05-Adjoint-Gradients/Discretize-then-Optimize-vs-Optimize-then-Discretize)
