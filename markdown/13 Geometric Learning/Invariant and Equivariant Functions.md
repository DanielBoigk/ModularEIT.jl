---
tags: [geometric-learning]
aliases: [Equivariance, Invariance]
---

Let a group $G$ act on spaces $X$ and $Y$ through representations $\rho_X(g)$ and $\rho_Y(g)$. A map $f:X\to Y$ is

- **invariant** if $f(\rho_X(g)x) = f(x)$ for all $g\in G$, $x\in X$;
- **equivariant** if $f(\rho_X(g)x) = \rho_Y(g)f(x)$ for all $g\in G$, $x\in X$.

Invariance is the special case of equivariance with the trivial output representation.

```tikz
\begin{document}
\begin{tikzpicture}
  \node (a) at (0,0) {$X$};
  \node (b) at (3,0) {$X$};
  \node (c) at (0,-2) {$Y$};
  \node (d) at (3,-2) {$Y$};
  \draw[->] (a) -- node[above] {$\rho_X(g)$} (b);
  \draw[->] (c) -- node[below] {$\rho_Y(g)$} (d);
  \draw[->] (a) -- node[left] {$f$} (c);
  \draw[->] (b) -- node[right] {$f$} (d);
\end{tikzpicture}
\end{document}
```

**Closure properties.**

- Compositions of equivariant maps are equivariant: $f(h(gx)) = f(g\,h(x)) = g\,f(h(x))$. So a network built from equivariant layers is equivariant.
- Equivariant linear maps form a vector space (0 and the identity are equivariant), which is closed under limits.
- Pointwise nonlinearities are equivariant for permutation representations, for example for $G$ acting on pixel positions.

**Why build it in.** If the target map is known to be equivariant (see [[Symmetries of the EIT Problem]]), restricting a network to equivariant functions reduces the number of free parameters, improves sample efficiency and generalisation, and guarantees consistent behaviour under the symmetry. Equivariant functions can be obtained by averaging (see [[Reynolds Operator]]) or by constraining the layers (see [[Equivariant Convolutions]]). Order matters: making features invariant too early discards orientation information that later layers may need. Typical architectures therefore use equivariant layers first and an invariant pooling at the end.

## References

1. T. Cohen, M. Welling (2016). *Group Equivariant Convolutional Networks*. ICML 2016. [arXiv:1602.07576](https://arxiv.org/abs/1602.07576)
2. M. M. Bronstein, J. Bruna, T. Cohen, P. Veličković (2021). *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges*. [arXiv:2104.13478](https://arxiv.org/abs/2104.13478)
3. M. Weiler, P. Forré, E. Verlinde, M. Welling (2023). *Equivariant and Coordinate Independent Convolutional Networks*. [maurice-weiler.gitlab.io/cnn_book](https://maurice-weiler.gitlab.io/cnn_book/EquivariantAndCoordinateIndependentCNNs.pdf)
