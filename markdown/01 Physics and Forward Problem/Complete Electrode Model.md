---
tags: [forward-problem, electrodes]
aliases: [CEM]
---

The **Complete Electrode Model (CEM)** is the standard forward model for real EIT measurements (see [[Electrode Models]]). With $L$ electrodes $e_\ell\subset\partial\Omega$, contact impedances $z_\ell>0$, injected currents $I\in\mathbb R^L$ and unknown electrode voltages $U\in\mathbb R^L$, find $(u,U)$ with

$$
\begin{aligned}
\nabla\cdot(\gamma\nabla u) &= 0 && \text{in }\Omega,\\
u + z_\ell\,\gamma\,\partial_\nu u &= U_\ell && \text{on } e_\ell,\ \ell=1,\dots,L,\\
\int_{e_\ell}\gamma\,\partial_\nu u\,\mathrm ds &= I_\ell && \ell = 1,\dots,L,\\
\gamma\,\partial_\nu u &= 0 && \text{on }\partial\Omega\setminus\textstyle\bigcup_\ell e_\ell .
\end{aligned}
$$

```tikz
\begin{document}
\begin{tikzpicture}
  \draw[thick] (-3,0) -- (3,0);
  \fill[black!70] (-1,0) rectangle (1,0.25);
  \node[above] at (0,0.3) {electrode $e_\ell$ with voltage $U_\ell$};
  \draw (0,0) -- (0,-0.2);
  \draw (0,-0.2) -- (0.15,-0.28) -- (-0.15,-0.44) -- (0.15,-0.6) -- (-0.15,-0.76) -- (0,-0.84);
  \draw (0,-0.84) -- (0,-1);
  \node[right] at (0.25,-0.5) {contact impedance $z_\ell$};
  \node at (0,-1.4) {$\Omega$, potential $u$};
\end{tikzpicture}
\end{document}
```

**Well-posedness.** Under charge conservation $\sum_\ell I_\ell = 0$ and a grounding condition $\sum_\ell U_\ell = 0$, the problem has a unique solution $(u,U)\in H^1(\Omega)\oplus\mathbb R^L$. The weak form uses the coercive bilinear form

$$
B\big((u,U),(v,V)\big) = \int_\Omega\gamma\nabla u\cdot\nabla v\,\mathrm dx + \sum_{\ell=1}^L\frac1{z_\ell}\int_{e_\ell}(u-U_\ell)(v-V_\ell)\,\mathrm ds = \sum_\ell I_\ell V_\ell .
$$

The resulting *electrode NtD matrix* $I\mapsto U$ is symmetric positive definite on $\{\sum I_\ell=0\}$. It is the finite-dimensional analogue of the [[Neumann-to-Dirichlet Map]]. As electrodes become small and numerous, the CEM approaches the continuum model. As $z_\ell\to 0$ it becomes the [[Shunt Model]]; for large $z_\ell$ the current density under the electrode becomes uniform, as in the [[Gap Model]].

Integrating the second condition over $e_\ell$ gives $U_\ell = |e_\ell|^{-1}\int_{e_\ell}u\,\mathrm ds + z_\ell I_\ell/|e_\ell|$: the voltage of a current-carrying electrode includes the contact voltage drop (see [[Measurement Protocols]]). The finite element system is derived in [[Discrete Electrode Models]].

## References

1. E. Somersalo, M. Cheney, D. Isaacson (1992). *Existence and Uniqueness for Electrode Models for Electric Current Computed Tomography*. SIAM J. Appl. Math. 52(4), 1023–1040. [doi:10.1137/0152060](https://doi.org/10.1137/0152060)
2. K.-S. Cheng, D. Isaacson, J. C. Newell, D. G. Gisser (1989). *Electrode models for electric current computed tomography*. IEEE Trans. Biomed. Eng. 36(9), 918–924. [doi:10.1109/10.35300](https://doi.org/10.1109/10.35300)
3. N. Hyvönen (2004). *Complete Electrode Model of Electrical Impedance Tomography: Approximation Properties and Characterization of Inclusions*. SIAM J. Appl. Math. 64(3), 902–931. [doi:10.1137/S0036139903423303](https://doi.org/10.1137/S0036139903423303)
