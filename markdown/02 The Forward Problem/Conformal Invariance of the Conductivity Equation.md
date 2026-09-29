---
tags: [physics, forward-problem, geometry]
aliases: [Conformal invariance, Conformal pull-back of EIT, Transformation of conductivities]
---

In two dimensions the [[Conductivity Equation]] is invariant under conformal maps. Every simply connected domain is the conformal image of the unit disk, so EIT on any such domain can be computed on the disk, with an isotropic conductivity, and mapped back.

## Invariance of the Dirichlet energy

Let $\Phi:D\to\Omega$ be conformal (holomorphic with $\Phi'\ne0$) and $v = u\circ\Phi$. By the Cauchy–Riemann equations $D\Phi$ is a rotation times $|\Phi'|$, so $|\nabla v|^2 = |\nabla u\circ\Phi|^2\,|\Phi'|^2$ and $\mathrm d x_\Omega = |\Phi'|^2\,\mathrm d x_D$. Hence

$$
\int_\Omega\sigma\,|\nabla u|^2\,\mathrm dx = \int_D(\sigma\circ\Phi)\,|\nabla v|^2\,\mathrm dx ,
$$

and by the [[Dirichlet and Thomson Principles|Dirichlet principle]] $u$ solves $\nabla\cdot\sigma\nabla u = 0$ in $\Omega$ exactly when $v$ solves $\nabla\cdot\tilde\sigma\nabla v = 0$ in $D$, with

$$
\tilde\sigma = \sigma\circ\Phi .
$$

The pulled-back conductivity stays **isotropic**, and its values are just transported. In higher dimensions nothing comparable holds: by Liouville's theorem the only conformal maps in $d\ge3$ are Möbius transformations.

## General diffeomorphisms

For a diffeomorphism that is not conformal, the same computation gives the **anisotropic** conductivity

$$
\tilde\sigma = |\det D\Phi|\,(D\Phi)^{-1}\,\sigma\,(D\Phi)^{-\top}\circ\Phi .
$$

Its anisotropy is measured by the distortion $K = \max|D\Phi|^2/|\det D\Phi|$, which is $K\ge1$ with equality exactly for conformal maps. This is the mechanism behind the non-uniqueness of [[Anisotropic Conductivities]]: diffeomorphisms that fix the boundary leave all boundary data unchanged.

## Boundary data and electrodes

On the boundary, arc length transforms as $\mathrm ds_\Omega = |\Phi'|\,\mathrm ds_D$. Therefore:

- **Voltages** are transported: $v = u\circ\Phi$ on $\partial D$.
- **Current densities** scale with the length factor: $\tilde\sigma\,\partial_\nu v = (g\circ\Phi)\,|\Phi'|$. The current through any boundary piece, in particular every electrode current $I_\ell = \int_{e_\ell}g\,\mathrm ds$, is invariant.
- In the [[Complete Electrode Model]], the contact term $\int_{e_\ell}(u-U_\ell)^2/z_\ell\,\mathrm ds$ becomes $\int_{\tilde e_\ell}(v-U_\ell)^2\,|\Phi'|/z_\ell\,\mathrm ds$. The pulled-back contact impedance $z_\ell/|\Phi'|$ varies along the electrode.
- In the [[Gap Model]], the uniform current density $I_\ell/|e_\ell|$ becomes $I_\ell|\Phi'|/|e_\ell|$, and mean voltages are weighted by $|\Phi'|$.

The [[Neumann-to-Dirichlet Map]] of $(\Omega,\sigma)$ and the one of $(D,\tilde\sigma)$ are therefore related by these weights. Electrode data, meaning currents in and electrode voltages out, are unchanged.

## Consequences

- **Model domains.** Forward and inverse problems on any simply connected domain can be solved on the disk: reconstruct $\tilde\sigma$ and push it forward by $\sigma = \tilde\sigma\circ\Phi^{-1}$. If the true boundary is only approximately known, reconstructing on a wrong domain produces an anisotropic error, and only its conformal part can be corrected.
- **Fast solvers.** A mesh of $\Omega$ obtained by mapping a disk mesh with $\Phi$ has a stiffness matrix that is spectrally equivalent to the one of the disk mesh, with constants close to $1$. Each element is mapped by an almost-similarity. Disk solvers therefore precondition the mapped problem (see [[Fast Solvers on Disk Domains]], [[Numerical Conformal Mapping]]).

**In ModularEIT.jl:** [`ConformalMap`](https://danielboigk.github.io/ModularEIT.jl/dev/api/linear_solvers/#ModularEIT.ConformalMap), [`conformal_grid`](https://danielboigk.github.io/ModularEIT.jl/dev/api/linear_solvers/#ModularEIT.conformal_grid).

## References

1. V. Kolehmainen, M. Lassas, P. Ola (2005). *The Inverse Conductivity Problem with an Imperfectly Known Boundary*. SIAM J. Appl. Math. 66(2), 365–383. [doi:10.1137/040612737](https://doi.org/10.1137/040612737)
2. J. L. Mueller, S. Siltanen (2012). *Linear and Nonlinear Inverse Problems with Practical Applications*. SIAM. [doi:10.1137/1.9781611972344](https://doi.org/10.1137/1.9781611972344)
3. K. Astala, L. Päivärinta, M. Lassas (2005). *Calderón's inverse problem for anisotropic conductivity in the plane*. Comm. Partial Differential Equations 30(1–2), 207–224. [doi:10.1081/PDE-200044485](https://doi.org/10.1081/PDE-200044485)
