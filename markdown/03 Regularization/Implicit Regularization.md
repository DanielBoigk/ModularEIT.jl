---
tags: [regularization]
aliases: [Regularization by discretization, Early stopping]
---

Not all regularisation appears as a penalty term. Several algorithmic choices stabilise EIT reconstructions implicitly:

- **Early stopping.** Iterative methods such as Landweber, Gauss–Newton and CG on the normal equations first fit the dominant, stable components and only later the unstable ones. Stopping by the discrepancy principle turns the iteration count into the regularisation parameter (see [[Choosing the Regularization Parameter]]).
- **Levenberg–Marquardt damping.** The term $\lambda L_{\text{LM}}$ in each Gauss–Newton step regularises the *step*, not the solution. It is a Tikhonov regularisation of the linearised problem (see [[Levenberg-Marquardt Method]]).
- **Few, low-frequency current patterns.** Using only low-frequency [[Current Patterns]], or the leading singular pairs (see [[Truncated SVD Regularization]]), discards the data components dominated by noise.
- **Discretisation.** Representing $\sigma$ on a coarse mesh or in a low-dimensional basis restricts it to a finite-dimensional set. By [[Stability of the Calderón Problem|Lipschitz stability]] in finite dimensions, this stabilises the problem, at the cost of resolution. Coarse-to-fine (multilevel) strategies refine the representation as the fit improves.
- **Box constraints.** Bounds $\sigma_{\min}\le\sigma\le\sigma_{\max}$ exclude the unbounded oscillations that minimising sequences of the unregularised problem tend to develop (see [[Box Constraints on Conductivity]]).

## References

1. H. W. Engl, M. Hanke, A. Neubauer (1996). *Regularization of Inverse Problems*. Kluwer. [doi:10.1007/978-94-009-1740-8](https://doi.org/10.1007/978-94-009-1740-8)
2. M. Hanke (1997). *A regularizing Levenberg–Marquardt scheme, with applications to inverse groundwater filtration problems*. Inverse Problems 13(1), 79–95. [doi:10.1088/0266-5611/13/1/007](https://doi.org/10.1088/0266-5611/13/1/007)
3. G. Alessandrini, S. Vessella (2005). *Lipschitz stability for the inverse conductivity problem*. Adv. Appl. Math. 35(2), 207–241. [doi:10.1016/j.aam.2004.12.002](https://doi.org/10.1016/j.aam.2004.12.002)
