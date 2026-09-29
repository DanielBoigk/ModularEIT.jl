---
title: ModularEIT Wiki
---

A wiki on the theory of **Electrical Impedance Tomography (EIT)**: imaging the electrical conductivity inside a body from currents and voltages measured at its surface. It covers the physics and mathematics of the forward and inverse problem, their discretisation and numerical solution, and classical and learned regularisation. It accompanies the Julia library [ModularEIT.jl](https://github.com/DanielBoigk/ModularEIT.jl), whose API is documented in the [API docs](https://danielboigk.github.io/ModularEIT.jl/dev/). Articles on implemented topics link to the corresponding functions.

Every article ends with its references, including DOI or arXiv links. Symbols are listed in [[Notation]].

## Where to start

- **New to EIT.** [[Electrical Impedance Tomography]] → [[Conductivity Equation]] → [[Electrode Models]] → [[Forward Map]] → [[Calderón Problem]] → [[Anatomy of an EIT Reconstruction]].
- **From the model to a simulation.** [[From Physics to Linear Algebra]] → [[Weak Formulation of the Conductivity Equation]] → [[Galerkin Method]] → [[Discrete Electrode Models]] → [[Choosing a Linear Solver]].
- **Reconstructing a conductivity.** [[Anatomy of an EIT Reconstruction]] → [[Variational Regularization]] → [[Adjoint State Method]] → [[Choosing an Optimizer]] → [[Noise Models for EIT Data]] → [[Inverse Crime]]. A worked example with code is the tutorial [Reconstructing a conductivity](https://danielboigk.github.io/ModularEIT.jl/dev/tutorials/reconstruction/).
- **Priors learned from data.** [[Classical and Learned Priors]] → [[Learned Regularization]] → [[Plug-and-Play Priors]] → [[Diffusion Models]] → [[Diffusion Models for EIT]].
- **Fast and adaptive computation.** [[Choosing a Linear Solver]] → [[Fast Solvers on Disk Domains]] → [[Numerical Conformal Mapping]] → [[Adaptive Meshing in EIT]].
- **The mathematics of uniqueness and stability.** [[Calderón Problem]] → [[Uniqueness in the Calderón Problem]] → [[Stability of the Calderón Problem]] → [[D-bar Method]].

## Topics

| | Topic | Contents |
|:--|:--|:--|
| 0 | [[00 Overviews/index\|Overviews]] | Short articles that connect the topics |
| 1 | [[01 Foundations of EIT/index\|Foundations of EIT]] | Physics, the conductivity equation, electrode models, measurement protocols |
| 2 | [[02 The Forward Problem/index\|The Forward Problem]] | Function spaces, well-posedness, boundary operators, conformal invariance |
| 3 | [[03 The Inverse Problem/index\|The Inverse Problem]] | Calderón problem, uniqueness, stability, D-bar, Bayesian view |
| 4 | [[04 Finite Elements/index\|Finite Elements]] | Galerkin discretisation, matrices, discrete electrode models, grounding |
| 5 | [[05 Linear Solvers/index\|Linear Solvers]] | Krylov methods, factorisations, multigrid, fast transform solvers |
| 6 | [[06 Meshes and Geometry/index\|Meshes and Geometry]] | Error estimation, adaptive refinement, conformal maps |
| 7 | [[07 Adjoint Gradients/index\|Adjoint Gradients]] | PDE-constrained optimisation, adjoint method, gradient representation |
| 8 | [[08 Regularization/index\|Regularization]] | Tikhonov, total variation, parameter choice |
| 9 | [[09 Optimization/index\|Optimization]] | Gauss–Newton, L-BFGS, proximal and splitting methods, stopping |
| 10 | [[10 Data and Noise/index\|Data and Noise]] | Synthetic data, noise models, inverse crime |
| 11 | [[11 Learned Priors/index\|Learned Priors]] | Plug-and-play, learned energies, implicit networks |
| 12 | [[12 Diffusion Models/index\|Diffusion Models]] | Score-based generative models and posterior sampling |
| 13 | [[13 Geometric Learning/index\|Geometric Learning]] | Symmetries and equivariant networks |
| 14 | [[14 Outlook/index\|Outlook]] | Software, design principles, open questions |

Every topic page gives a reading order through its articles and lists them all. The explorer on the left and the search find any article directly.
