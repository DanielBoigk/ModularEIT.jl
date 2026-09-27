---
tags: [forward-model]
aliases: [CEM]
---

The **Complete Electrode Model** (CEM) is the standard model for the
[[Forward Problem]] of [[Electrical Impedance Tomography]]. It accounts for the
finite size of the $L$ electrodes $e_1, \dots, e_L$ and for the contact impedances
$z_\ell > 0$ between electrode and skin.

Find the potential $u$ and electrode voltages $U \in \mathbb{R}^L$ such that

$$
\begin{aligned}
\nabla \cdot (\sigma \nabla u) &= 0 && \text{in } \Omega, \\
u + z_\ell \, \sigma \partial_\nu u &= U_\ell && \text{on } e_\ell, \\
\int_{e_\ell} \sigma \partial_\nu u \, \mathrm{d}s &= I_\ell && \ell = 1, \dots, L, \\
\sigma \partial_\nu u &= 0 && \text{on } \partial\Omega \setminus \textstyle\bigcup_\ell e_\ell .
\end{aligned}
$$

Uniqueness requires the grounding condition $\sum_\ell U_\ell = 0$ and conservation
of charge $\sum_\ell I_\ell = 0$.

## Electrode boundary condition

```tikz
\usetikzlibrary{decorations.pathmorphing}
\begin{document}
\begin{tikzpicture}
  \draw[thick] (-3,0) -- (3,0);
  \fill[black!70] (-1,0) rectangle (1,0.25);
  \node[above] at (0,0.3) {electrode $e_\ell$, voltage $U_\ell$};
  \draw[decorate, decoration={zigzag, amplitude=2pt, segment length=4pt}] (0,0) -- (0,-1);
  \node[right] at (0.2,-0.5) {$z_\ell$};
  \node at (0,-1.5) {$\Omega$, potential $u$};
\end{tikzpicture}
\end{document}
```

## Implementation

````tabs
tab: Julia
```julia
using ModularEIT
mesh = circle_mesh(32)
electrodes = ring_electrodes(mesh, 16; impedance = 1e-2)
problem = ForwardProblem(mesh, electrodes)
```
tab: Math
The mock solver in the package replaces the finite-element system by
a diagonal resistor network: U = I / mean(σ) + z .* I
````
