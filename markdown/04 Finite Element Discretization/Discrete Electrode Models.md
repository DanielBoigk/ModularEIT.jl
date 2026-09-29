---
tags: [numerics, fem, electrodes, forward-problem]
aliases: [Injection and measurement matrices, Discrete CEM]
---

After a Galerkin discretisation with basis functions $\varphi_i$, every [[Electrode Models|electrode model]] becomes a linear system with an **injection matrix** $P$ and a **measurement matrix** $Q$:

$$
A(\sigma)\,\mathbf x = P\,I,\qquad V = Q\,\mathbf x ,
$$

where $I$ collects the injected currents and $V$ the measured voltages. For all models except the CEM, $A(\sigma) = L_\sigma$ is the [[Weighted Stiffness Matrix]] and $\mathbf x = \mathbf u$ the nodal potential.

| model | $P$ (column $\ell$) | $Q$ (row $m$) |
|:--|:--|:--|
| continuum | $\int_{\partial\Omega}\varphi_i\varphi_j\,\mathrm ds$ restricted to boundary nodes ($I$ = density coefficients) | unit row of boundary node $m$ |
| [[Point Electrode Model\|point]] | unit vector of node $x_\ell$ | unit row of node $x_m$ |
| [[Gap Model\|gap]] | $\int_{e_\ell}\varphi_i\,\mathrm ds\,/\,\vert e_\ell\vert$ | $\int_{e_m}\varphi_j\,\mathrm ds\,/\,\vert e_m\vert$ |
| [[Complete Electrode Model\|CEM]] | unit vector of the unknown $U_\ell$ | unit row of $U_m$ |

Injection and measurement sites are independent: $P$ has one column per drive electrode, $Q$ one row per measurement electrode (see [[Measurement Protocols]]). For the gap model with the same electrodes, $Q = P^\top$, which is discrete reciprocity.

## Complete electrode model

The CEM adds the electrode voltages $U\in\mathbb R^L$ as unknowns. With the electrode mass matrices $(M_\ell)_{ij} = \int_{e_\ell}\varphi_i\varphi_j\,\mathrm ds$ and vectors $(d_\ell)_i = \int_{e_\ell}\varphi_i\,\mathrm ds$, the weak form of the [[Complete Electrode Model]] gives

$$
\begin{bmatrix} L_\sigma + \sum_\ell z_\ell^{-1}M_\ell & -D \\ -D^\top & \operatorname{diag}(|e_\ell|/z_\ell)\end{bmatrix}
\begin{bmatrix}\mathbf u\\ U\end{bmatrix} = \begin{bmatrix}0\\ I\end{bmatrix},\qquad D = \big[\,d_1/z_1\ \cdots\ d_L/z_L\,\big].
$$

The matrix is symmetric positive semidefinite. Its null space is spanned by the constant vector on $(\mathbf u, U)$ jointly, and the grounding $\sum_\ell U_\ell = 0$ fixes the constant. Only the upper-left block depends on $\sigma$. The contact terms form a constant matrix $A_0$, so $A(\sigma) = A_0 + L_\sigma$.

As $z_\ell\to\infty$, the current density under each electrode becomes uniform, and $U_\ell$ approaches the gap-model voltage plus $z_\ell I_\ell/|e_\ell|$.

## Voltage-driven (Dirichlet) problems

Every model also has a voltage-driven counterpart. Prescribe the unknowns on a set $B$ of Dirichlet degrees of freedom through an expansion matrix $E$, $\mathbf x_B = E\,U$, and solve for the free degrees of freedom $F$:

$$
A_{FF}\,\mathbf x_F = -A_{FB}\,E\,U,\qquad I = C^{-1}E^\top (A\,\mathbf x)_B,\qquad C = E^\top P_B .
$$

- continuum: $B$ = boundary nodes, $E = \mathrm{Id}$, $C$ = boundary mass matrix (the currents come out as densities);
- point: $B$ = electrode nodes, $E = \mathrm{Id}$, $C = \mathrm{Id}$;
- gap: $B$ = all nodes of the drive electrodes, $E$ = electrode indicator. This is the [[Shunt Model]], $C = \mathrm{Id}$;
- CEM: $B$ = the voltages $U$, $E = \mathrm{Id}$, $C = \mathrm{Id}$.

$(A\,\mathbf x)_B$ is the residual of the discrete equation on the Dirichlet nodes, the variationally consistent discrete normal current. The matrix $I\mapsto U$ inverts $U\mapsto I$ whenever both problems describe the same physics (continuum, point, CEM). The map $U\mapsto I$ is the Schur complement $A_{BB}-A_{BF}A_{FF}^{-1}A_{FB}$, a discrete [[Dirichlet-to-Neumann Map]] (see [[Discrete Boundary Operator]]).

**In ModularEIT.jl:** [`ForwardModel`](https://danielboigk.github.io/ModularEIT.jl/dev/api/forward/#ModularEIT.ForwardModel), [`ContinuumModel`](https://danielboigk.github.io/ModularEIT.jl/dev/api/forward/#ModularEIT.ContinuumModel), [`PointElectrodeModel`](https://danielboigk.github.io/ModularEIT.jl/dev/api/forward/#ModularEIT.PointElectrodeModel), [`GapModel`](https://danielboigk.github.io/ModularEIT.jl/dev/api/forward/#ModularEIT.GapModel), [`CompleteElectrodeModel`](https://danielboigk.github.io/ModularEIT.jl/dev/api/forward/#ModularEIT.CompleteElectrodeModel).

## References

1. E. Somersalo, M. Cheney, D. Isaacson (1992). *Existence and Uniqueness for Electrode Models for Electric Current Computed Tomography*. SIAM J. Appl. Math. 52(4), 1023–1040. [doi:10.1137/0152060](https://doi.org/10.1137/0152060)
2. P. J. Vauhkonen, M. Vauhkonen, T. Savolainen, J. P. Kaipio (1999). *Three-dimensional electrical impedance tomography based on the complete electrode model*. IEEE Trans. Biomed. Eng. 46(9), 1150–1160. [doi:10.1109/10.784147](https://doi.org/10.1109/10.784147)
3. J. P. Kaipio, V. Kolehmainen, E. Somersalo, M. Vauhkonen (2000). *Statistical inversion and Monte Carlo sampling methods in electrical impedance tomography*. Inverse Problems 16(5), 1487–1522. [doi:10.1088/0266-5611/16/5/321](https://doi.org/10.1088/0266-5611/16/5/321)
4. N. Polydorides, W. R. B. Lionheart (2002). *A Matlab toolkit for three-dimensional electrical impedance tomography: a contribution to the Electrical Impedance and Diffuse Optical Reconstruction Software project*. Meas. Sci. Technol. 13(12), 1871–1883. [doi:10.1088/0957-0233/13/12/310](https://doi.org/10.1088/0957-0233/13/12/310)
