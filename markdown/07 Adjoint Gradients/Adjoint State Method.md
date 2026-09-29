---
tags: [adjoint, gradient]
aliases: [Adjoint method, Adjoint-state method]
---

The **adjoint state method** computes the gradient of a functional $\hat J(\sigma) = J(u(\sigma))$, where $u(\sigma)$ solves a PDE. Its cost is one extra linear solve, no matter how many parameters $\sigma$ has.

**The problem it solves.** By the chain rule $\hat J'(\sigma)[\delta\sigma] = J'(u)[u'(\sigma)\delta\sigma]$. Computing $u'(\sigma)\delta\sigma$ needs one PDE solve *per direction* $\delta\sigma$, that is, per parameter. For $10^4$ pixels that is $10^4$ solves per gradient.

**The trick.** Using the [[Lagrangian Formulation|Lagrangian]] $\mathcal L(\sigma,u,\lambda) = J(u)+\langle\lambda,e(\sigma,u)\rangle$:

1. If $u = u(\sigma)$ solves the [[State Equation]], then $e(\sigma,u) = 0$ and $\mathcal L(\sigma,u(\sigma),\lambda) = \hat J(\sigma)$ for **every** $\lambda$.
2. Differentiate in $\sigma$:
   $$\hat J'(\sigma) = \partial_\sigma\mathcal L + \partial_u\mathcal L\circ u'(\sigma).$$
3. Choose $\lambda$ such that $\partial_u\mathcal L = 0$. This is the [[Adjoint Equation]]. Then the expensive term $u'(\sigma)$ drops out:
   $$\hat J'(\sigma) = \partial_\sigma\mathcal L(\sigma,u,\lambda).$$

**Recipe for EIT** (per current pattern $i$):

1. solve the state equation $L_\sigma u_i = g_i$;
2. solve the adjoint equation $L_\sigma\lambda_i = \partial_uJ_i$, with the *same* matrix, since the problem is self-adjoint;
3. evaluate $\nabla\hat J = \sum_i -\nabla u_i\cdot\nabla\lambda_i$ (see [[Functional Derivative of the Data Misfit]]).

Two solves per pattern give the full gradient. With a regulariser, add $\beta\nabla\mathcal R(\sigma)$. The state and adjoint equations are unaffected because $\mathcal R$ does not depend on $u$.

```tikz
\begin{document}
\begin{tikzpicture}[node distance=2.3cm, every node/.style={draw, rounded corners, minimum width=2cm, minimum height=0.8cm, align=center}]
  \node (s) {$\sigma$};
  \node[right of=s, xshift=0.6cm] (u) {state $u$};
  \node[right of=u, xshift=0.6cm] (J) {misfit $J(u)$};
  \node[below of=u, yshift=0.8cm] (l) {adjoint $\lambda$};
  \node[below of=s, yshift=0.8cm] (g) {$\nabla J=-\nabla u\cdot\nabla\lambda$};
  \draw[->] (s) -- (u);
  \draw[->] (u) -- (J);
  \draw[->] (J) |- (l);
  \draw[->] (l) -- (g);
  \draw[->] (u) -- (g);
\end{tikzpicture}
\end{document}
```

This is the continuous analogue of reverse-mode automatic differentiation (see [[Automatic Differentiation vs Adjoint Methods]]).

**In ModularEIT.jl:** [`AdjointStateObjective`](https://danielboigk.github.io/ModularEIT.jl/dev/api/objectives/#ModularEIT.AdjointStateObjective), [`value_and_gradient!`](https://danielboigk.github.io/ModularEIT.jl/dev/api/objectives/#ModularEIT.value_and_gradient!).

## References

1. R.-E. Plessix (2006). *A review of the adjoint-state method for computing the gradient of a functional with geophysical applications*. Geophys. J. Int. 167(2), 495–503. [doi:10.1111/j.1365-246X.2006.02978.x](https://doi.org/10.1111/j.1365-246X.2006.02978.x)
2. M. Hinze, R. Pinnau, M. Ulbrich, S. Ulbrich (2009). *Optimization with PDE Constraints*. Springer. [doi:10.1007/978-1-4020-8839-1](https://doi.org/10.1007/978-1-4020-8839-1)
3. A. M. Bradley (2024). *PDE-constrained optimization and the adjoint method*. Lecture notes, Stanford. [cs.stanford.edu/~ambrad/adjoint_tutorial.pdf](https://cs.stanford.edu/~ambrad/adjoint_tutorial.pdf)
4. D. Lahaye, W. Mulckhuyse (2012). *Adjoint sensitivity in PDE constrained least squares problems as a multiphysics problem*. COMPEL 31(3), 895–903. [doi:10.1108/03321641211209780](https://doi.org/10.1108/03321641211209780)
5. F. J. Margotti (2015). *On Inexact Newton Methods for Inverse Problems in Banach Spaces*. PhD thesis, KIT. [doi:10.5445/IR/1000048606](https://doi.org/10.5445/IR/1000048606)
