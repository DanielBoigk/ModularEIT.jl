---
tags: [overview]
aliases: [EIT]
---

**Electrical Impedance Tomography (EIT)** is an imaging method that recovers the electrical conductivity $\gamma$ inside a body $\Omega$ from measurements made only on its boundary $\partial\Omega$.

Electrodes are attached around the boundary. Known currents are driven through some electrodes and the resulting voltages are recorded on all of them. Repeating this for many current patterns gives a set of boundary *voltage–current pairs*. The aim is to infer the interior conductivity that is consistent with all of them.

```tikz
\begin{document}
\begin{tikzpicture}
  \draw[thick] (0,0) circle (2);
  \fill[gray!30] (0.7,0.5) circle (0.6);
  \node at (0.7,0.5) {$\gamma_1$};
  \node at (-0.8,-0.6) {$\gamma_0$};
  \foreach \a in {0,22.5,...,337.5} { \fill (\a:2) circle (0.09); }
  \draw[->, thick] (-3.3,0.6) -- (-2.15,0.2) node[pos=0, above] {$I$};
  \draw[<-, thick] (-3.3,-0.6) -- (-2.15,-0.2) node[pos=0, below] {$I$};
  \node at (2.9,1.6) {$\partial\Omega$};
  \node at (-0.9,1.1) {$\Omega$};
\end{tikzpicture}
\end{document}
```

Mathematically the problem splits into two parts:

- the [[Forward Map|forward problem]]: given $\gamma$, predict the boundary data. It is governed by the [[Conductivity Equation]] and encoded in the [[Dirichlet-to-Neumann Map]];
- the inverse problem, the [[Calderón Problem]]: given the boundary data, recover $\gamma$. It is severely [[Well-Posedness|ill-posed]] (see [[Stability of the Calderón Problem]]), so practical reconstructions need [[Variational Regularization|regularization]].

Real devices only see finitely many electrodes and have contact impedances, which the [[Complete Electrode Model]] accounts for. Many analyses and simulations instead use the idealised *continuum model*, where every point of $\partial\Omega$ is available for injecting current and measuring voltage (see [[Electrode Models]]).

Compared with other [[Tomographic Imaging Modalities]], EIT is cheap, fast, portable and uses no ionising radiation, but its spatial resolution is low. See [[Applications of EIT]].

## References

1. M. Cheney, D. Isaacson, J. C. Newell (1999). *Electrical Impedance Tomography*. SIAM Review 41(1), 85–101. [doi:10.1137/S0036144598333613](https://doi.org/10.1137/S0036144598333613)
2. L. Borcea (2002). *Electrical impedance tomography*. Inverse Problems 18(6), R99–R136. [doi:10.1088/0266-5611/18/6/201](https://doi.org/10.1088/0266-5611/18/6/201)
3. D. S. Holder (ed.) (2004). *Electrical Impedance Tomography: Methods, History and Applications*. CRC Press. [doi:10.1201/9781420034462](https://doi.org/10.1201/9781420034462)
4. J. L. Mueller, S. Siltanen (2012). *Linear and Nonlinear Inverse Problems with Practical Applications*. SIAM. [doi:10.1137/1.9781611972344](https://doi.org/10.1137/1.9781611972344)
