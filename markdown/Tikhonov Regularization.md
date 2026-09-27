---
tags: [regularization]
---

**Tikhonov regularization** penalises the distance to a reference conductivity
$\sigma_0$:

$$
R(\sigma) = \frac{\alpha}{2} \lVert \sigma - \sigma_0 \rVert_2^2 .
$$

It makes the [[Electrical Impedance Tomography]] problem stable but produces smooth,
blurred inclusions. For piecewise-constant targets prefer [[Total Variation]].

## Linearised solution

Linearising the [[Forward Problem]] around $\sigma_0$ with Jacobian $J$ gives the
closed form update

$$
\delta\sigma = (J^\top J + \alpha I)^{-1} J^\top (U^\delta - F(\sigma_0)).
$$

## Choosing $\alpha$

```tikz
\begin{document}
\begin{tikzpicture}[scale=1.2]
  \draw[->] (0,0) -- (4.2,0) node[right] {$\log \|F(\sigma_\alpha) - U\|$};
  \draw[->] (0,0) -- (0,3.2) node[above] {$\log \|\sigma_\alpha\|$};
  \draw[thick] (0.4,3) .. controls (0.5,0.6) and (0.8,0.5) .. (4,0.4);
  \fill (0.62,0.9) circle (0.06) node[right] {optimal $\alpha$};
\end{tikzpicture}
\end{document}
```

The *L-curve* above plots residual against solution norm; the corner is a common
heuristic for the regularisation parameter.

In code: `Tikhonov(α; σ₀ = 1.0)`.
