---
tags: [forward-problem, measurement]
aliases: [Boundary patterns, Fourier current patterns]
---

A **current pattern** is a boundary current $g$ with $\int_{\partial\Omega} g\,\mathrm ds = 0$ (or electrode currents with $\sum_\ell I_\ell = 0$). The data set is a collection of pairs $(g_i, f_i)$ with $f_i = \mathcal R_\gamma g_i$ (see [[Neumann-to-Dirichlet Map]]). With $N$ linearly independent patterns one observes the NtD map on an $N$-dimensional subspace.

**Common choices**

- *Adjacent / neighbouring*: current between neighbouring electrodes. Simple hardware, but poor sensitivity in the centre.
- *Opposite* and *skip-$k$* patterns: current between electrodes a fixed distance apart.
- *Trigonometric (Fourier) patterns*: on a boundary parametrised by the angle or arc length $\theta$,
  $$ g_k(\theta) = \cos(k\theta),\ \ \sin(k\theta),\qquad k = 1,2,\dots $$
  They are orthogonal, zero-mean and easy to build.

**Decay with frequency.** For a homogeneous disc, $\mathcal R_1 e^{ik\theta} = |k|^{-1}e^{ik\theta}$, so the voltage amplitude falls off like $1/k$. The information a pattern carries about the interior decays even faster. A harmonic of frequency $k$ behaves like $r^{|k|}e^{ik\theta}$ in polar coordinates, so high-frequency patterns only probe a thin layer near the boundary. With noise, only the first few frequencies are informative (see [[Decay of Boundary Measurements]]).

**Optimal patterns.** Isaacson (1986) showed that the patterns that best *distinguish* two conductivities $\gamma_1,\gamma_2$ are the eigenfunctions of $\mathcal R_{\gamma_1}-\mathcal R_{\gamma_2}$ with the largest eigenvalues. For rotationally symmetric perturbations of a disc, these are exactly the trigonometric patterns. In practice one can compute them from data through the SVD of an estimated boundary operator (see [[Truncated SVD Regularization]]).

**In ModularEIT.jl:** [`trigonometric_patterns`](https://danielboigk.github.io/ModularEIT.jl/dev/api/forward/#ModularEIT.trigonometric_patterns), [`pattern_svd`](https://danielboigk.github.io/ModularEIT.jl/dev/api/forward/#ModularEIT.pattern_svd).

## References

1. D. Isaacson (1986). *Distinguishability of Conductivities by Electric Current Computed Tomography*. IEEE Trans. Med. Imaging 5(2), 91–95. [doi:10.1109/TMI.1986.4307752](https://doi.org/10.1109/TMI.1986.4307752)
2. D. Gisser, D. Isaacson, J. C. Newell (1990). *Electric Current Computed Tomography and Eigenvalues*. SIAM J. Appl. Math. 50(6), 1623–1634. [doi:10.1137/0150096](https://doi.org/10.1137/0150096)
3. J. L. Mueller, S. Siltanen (2012). *Linear and Nonlinear Inverse Problems with Practical Applications*. SIAM. [doi:10.1137/1.9781611972344](https://doi.org/10.1137/1.9781611972344)
