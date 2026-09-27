---
tags: [data, ill-posedness]
---

Why can only a few current patterns be used effectively?

**Voltage amplitude.** For a homogeneous unit disc, a Fourier current $g_k = \cos(k\theta)$ produces the boundary voltage $f_k = \frac1{k}\cos(k\theta)$. The [[Neumann-to-Dirichlet Map]] has eigenvalues $1/|k|$. For general conductivities the NtD map is a pseudodifferential operator of order $-1$, so the same $1/k$ decay holds asymptotically. Plotting $\|f_k\|$ against $k$ for any conductivity shows this monotone decay: low frequencies carry most of the voltage.

**Information content.** More important is how the *difference* $\mathcal R_\sigma-\mathcal R_{\sigma_0}$, the part that carries information about the interior, decays. The harmonic extension of $\cos(k\theta)$ is $r^k\cos(k\theta)$. It is concentrated in a boundary layer of width $\sim1/k$, so an inclusion at depth $d$ changes the $k$-th measurement by roughly $(1-d)^{2k}$. That is **exponentially** small in $k$. The singular values of the linearised map from conductivity to data therefore decay exponentially. This is the discrete face of the logarithmic [[Stability of the Calderón Problem|stability]].

**Consequence.** Once $(1-d)^{2k}$ falls below the relative noise level, higher patterns add only noise. Truncating to the leading patterns or singular pairs is a natural regulariser (see [[Truncated SVD Regularization]] and [[Current Patterns]]).

## References

1. D. Isaacson (1986). *Distinguishability of Conductivities by Electric Current Computed Tomography*. IEEE Trans. Med. Imaging 5(2), 91–95. [doi:10.1109/TMI.1986.4307752](https://doi.org/10.1109/TMI.1986.4307752)
2. J. L. Mueller, S. Siltanen (2012). *Linear and Nonlinear Inverse Problems with Practical Applications*. SIAM. [doi:10.1137/1.9781611972344](https://doi.org/10.1137/1.9781611972344)
3. N. Mandache (2001). *Exponential instability in an inverse problem for the Schrödinger equation*. Inverse Problems 17(5), 1435–1444. [doi:10.1088/0266-5611/17/5/313](https://doi.org/10.1088/0266-5611/17/5/313)
