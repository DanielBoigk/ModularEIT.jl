---
tags: [outlook]
---

Research directions that come out of the topics in this wiki.

**Learned priors without hallucination.** How can a generative prior be restricted to the directions that the data cannot see? One idea is a proximal step whose metric is built from the forward operator, so that changes which alter the predicted currents are penalised, while changes in the null space of the data are left to the prior (see [[Hallucinations and Uncertainty]] and [[Diffusion Proximal Operator]]).

**Learned metrics.** Can [[Input Convex Neural Networks]] or optimal-transport-based distances provide [[Data Fidelity Terms]] with fewer spurious local minima than boundary $L^2$?

**Optimal experiments.** Which current patterns or frequencies should be applied, possibly adaptively as the reconstruction proceeds? This is especially relevant for complex, multi-frequency EIT (see [[Complex Conductivity]] and [[Current Patterns]]). It is a problem of sequential optimal experimental design, for which reinforcement learning is one possible tool.

**Symmetry and geometry.** What is the full symmetry group of the discrete EIT problem, including conformal maps in 2D (see [[Symmetries of the EIT Problem]])? Can it be turned into geometry-aware regularisers or metrics that are cheap to evaluate?

**Better optimisers.** Can stochastic dynamical systems, for example stochastic pattern-wise updates or Langevin-type samplers, replace Gauss–Newton and L-BFGS? Such methods would combine speed, parallelism and uncertainty quantification.

**Uncertainty quantification.** Beyond re-running samplers with different seeds, how can one calibrate uncertainty for learned-prior EIT reconstructions (see [[Bayesian Inversion]])?

## References

1. A. Alexanderian (2021). *Optimal experimental design for infinite-dimensional Bayesian inverse problems governed by PDEs: a review*. Inverse Problems 37(4), 043001. [doi:10.1088/1361-6420/abe10c](https://doi.org/10.1088/1361-6420/abe10c)
2. G. Bao, Y. Zhang (2022). *Optimal Transportation for Electrical Impedance Tomography*. [arXiv:2210.16082](https://arxiv.org/abs/2210.16082)
3. B. Amos, L. Xu, J. Z. Kolter (2017). *Input Convex Neural Networks*. ICML 2017. [arXiv:1609.07152](https://arxiv.org/abs/1609.07152)
