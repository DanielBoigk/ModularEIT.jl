---
title: ModularEIT Wiki
---

A wiki on the theory of **Electrical Impedance Tomography (EIT)**: the physics and mathematics of the forward and inverse problem, finite element discretisation, adjoint-based gradients, optimisation, and classical and learned regularisation. It accompanies the Julia library [ModularEIT.jl](https://github.com/DanielBoigk/ModularEIT.jl), whose API is documented in the [Documenter docs](https://danielboigk.github.io/ModularEIT.jl/dev/).

Every article ends with its references, including DOI or arXiv links. Symbols are listed in [[Notation]].

## 1. Physics and forward problem
[[Electrical Impedance Tomography]] · [[Tomographic Imaging Modalities]] · [[Applications of EIT]] · [[Ohm's Law in Continuum Form]] · [[Continuity Equation]] · [[Quasi-Static Approximation]] · [[Complex Conductivity]] · [[Conductivity Equation]] · [[Sobolev and Trace Spaces]] · [[Dirichlet Problem]] · [[Neumann Problem]] · [[Dirichlet-to-Neumann Map]] · [[Neumann-to-Dirichlet Map]] · [[Properties of the Boundary Operators]] · [[Forward Map]] · [[Dirichlet and Thomson Principles]] · [[Electrode Models]] · [[Point Electrode Model]] · [[Gap Model]] · [[Shunt Model]] · [[Complete Electrode Model]] · [[Measurement Protocols]] · [[Current Patterns]]

## 2. Inverse problem
[[Calderón Problem]] · [[Well-Posedness]] · [[Uniqueness in the Calderón Problem]] · [[Boundary Determination]] · [[Complex Geometrical Optics Solutions]] · [[Anisotropic Conductivities]] · [[Stability of the Calderón Problem]] · [[Linearized EIT and the Sensitivity Kernel]] · [[Reconstruction Algorithms with Guarantees]] · [[D-bar Method]] · [[Bayesian Inversion]] · [[EIT Software and Solvers]] · [[Deep Learning for EIT]]

## 3. Regularisation
[[Variational Regularization]] · [[Data Fidelity Terms]] · [[Choosing the Regularization Parameter]] · [[Tikhonov Regularization]] · [[Total Variation]] · [[Smoothed Total Variation]] · [[Truncated SVD Regularization]] · [[Implicit Regularization]]

## 4. Finite element discretisation
[[Galerkin Method]] · [[Weak Formulation of the Conductivity Equation]] · [[Lax-Milgram Theorem]] · [[Green's Identities]] · [[Lagrange Finite Elements]] · [[Mass Matrix]] · [[Stiffness Matrix]] · [[Weighted Stiffness Matrix]] · [[Conductivity Tensor]] · [[Numerical Quadrature and Assembly]] · [[Enforcing Dirichlet Conditions]] · [[Discrete Electrode Models]] · [[Null Space of the Neumann Problem]] · [[Boundary Mass and Stiffness Matrices]] · [[Discrete Fractional Sobolev Norms]] · [[L2 Projection]] · [[Discrete Boundary Operator]] · [[Conjugate Gradient Method]] · [[Projected Conjugate Gradient]] · [[Block Conjugate Gradient]] · [[Projected Cholesky Factorization]] · [[Grounding of the Potential]] · [[MINRES]] · [[LSQR]] · [[Algebraic Multigrid]] · [[Block Krylov Methods]] · [[A Posteriori Error Estimation and Adaptive Meshing]] · [[Adaptive Meshing in EIT]] · [[Zienkiewicz-Zhu Estimator]] · [[Goal-Oriented Error Estimation]] · [[Dörfler Marking]] · [[Hanging Nodes]] · [[Newest Vertex Bisection]]

## 5. Adjoint gradients
[[Iterative Reconstruction Loop]] · [[PDE-Constrained Optimization]] · [[Lagrangian Formulation]] · [[KKT Conditions]] · [[Adjoint State Method]] · [[State Equation]] · [[Adjoint Equation]] · [[Adjoint Method for the Dirichlet Problem]] · [[Functional Derivative of the Data Misfit]] · [[Kohn-Vogelius Functional]] · [[Gradient Representation and the Riesz Map]] · [[Automatic Differentiation vs Adjoint Methods]] · [[Discretize-then-Optimize vs Optimize-then-Discretize]] · [[Rules of the Calculus of Variations]] · [[Gradient Testing]]

## 6. Optimisation
[[Gauss-Newton Method]] · [[Levenberg-Marquardt Method]] · [[Line Search]] · [[L-BFGS]] · [[L-BFGS-B]] · [[Proximal Operator]] · [[ADMM]] · [[Chambolle-Pock Algorithm]] · [[Nested ADMM Reconstruction]] · [[Box Constraints on Conductivity]] · [[Stopping Criteria]] · [[Stochastic and Adaptive Gradient Methods]]

## 7. Data and noise
[[Synthetic Conductivity Data]] · [[Decay of Boundary Measurements]] · [[Noise Models for EIT Data]] · [[Inverse Crime]] · [[Spectral Image Corruption]]

## 8. Learned priors
[[Learned Regularization]] · [[Plug-and-Play Priors]] · [[Regularization by Denoising]] · [[Manifold Hypothesis]] · [[Neural ODEs]] · [[Deep Equilibrium Models]] · [[Energy-Based Models]] · [[Langevin Dynamics]] · [[Input Convex Neural Networks]] · [[U-Net]] · [[Hallucinations and Uncertainty]]

## 9. Diffusion models
[[Diffusion Models]] · [[DDPM Forward Process]] · [[Noise Schedule]] · [[Variance-Preserving SDE]] · [[Continuous Limit of the DDPM Chain]] · [[Score Function]] · [[Denoising Score Matching]] · [[Reverse-Time SDE]] · [[Probability Flow ODE]] · [[Euler-Maruyama Method]] · [[DDPM Ancestral Sampling]] · [[Tweedie's Formula]] · [[Sinusoidal Time Embedding]] · [[Diffusion Posterior Sampling]] · [[DiffPIR]] · [[RED-Diff]] · [[Diffusion Proximal Operator]] · [[Diffusion Models for EIT]]

## 10. Geometric learning
[[Symmetries of the EIT Problem]] · [[Invariant and Equivariant Functions]] · [[Dihedral Group D4]] · [[Reynolds Operator]] · [[Invariant Filter Banks]] · [[Equivariant Convolutions]]

## 11. Outlook
[[Design Principles for EIT Solvers]] · [[Open Questions]]
