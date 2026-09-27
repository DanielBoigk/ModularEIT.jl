---
tags: [regularization]
aliases: [Regularization parameter, L-curve, Discrepancy principle]
---

The parameter $\beta$ in a [[Variational Regularization|variational objective]] balances data fit against prior. If it is too small, the reconstruction fits the noise; if it is too large, detail is lost. Standard selection rules:

**Morozov's discrepancy principle.** Choose the largest $\beta$ with

$$
\|\mathcal F(\sigma_\beta) - y^\delta\| \le \tau\,\delta ,\qquad \tau > 1 \text{ fixed},
$$

where $\delta$ is the known noise level: do not fit the data better than the noise allows. It is a regularisation strategy with convergence guarantees. The same rule stops iterative methods ([[Levenberg-Marquardt Method|regularising Levenberg–Marquardt]], Landweber) early.

**L-curve.** Plot $\log\|\mathcal F(\sigma_\beta)-y^\delta\|$ against $\log\mathcal R(\sigma_\beta)$ for many $\beta$. The curve typically looks like an "L", and its corner (maximum curvature) is a heuristic choice that needs no noise level.

```tikz
\begin{document}
\begin{tikzpicture}[scale=1.1]
  \draw[->] (0,0) -- (4.3,0) node[below left] {$\log\|F(\sigma_\beta)-y\|$};
  \draw[->] (0,0) -- (0,3.3) node[above] {$\log R(\sigma_\beta)$};
  \draw[thick] (0.4,3) .. controls (0.5,0.6) and (0.8,0.5) .. (4,0.4);
  \fill (0.64,0.88) circle (0.06) node[right] {corner};
  \node[left] at (0.45,2.6) {small $\beta$};
  \node[above] at (3.6,0.45) {large $\beta$};
\end{tikzpicture}
\end{document}
```

**Generalised cross-validation (GCV)** minimises a leave-one-out prediction error estimate. It needs no noise level.

**Continuation.** In practice $\beta$ is often decreased gradually during the iterations. This also helps nonlinear solvers avoid poor local minima.

## References

1. H. W. Engl, M. Hanke, A. Neubauer (1996). *Regularization of Inverse Problems*. Kluwer. [doi:10.1007/978-94-009-1740-8](https://doi.org/10.1007/978-94-009-1740-8)
2. P. C. Hansen (1992). *Analysis of Discrete Ill-Posed Problems by Means of the L-Curve*. SIAM Review 34(4), 561–580. [doi:10.1137/1034115](https://doi.org/10.1137/1034115)
3. G. H. Golub, M. Heath, G. Wahba (1979). *Generalized Cross-Validation as a Method for Choosing a Good Ridge Parameter*. Technometrics 21(2), 215–223. [doi:10.1080/00401706.1979.10489751](https://doi.org/10.1080/00401706.1979.10489751)
