---
tags: [inverse-problem, reconstruction]
aliases: [Dbar method, D-bar]
---

The **D-bar method** is a direct, non-iterative reconstruction algorithm for 2D EIT. It is based on Nachman's (1996) constructive uniqueness proof (see [[Reconstruction Algorithms with Guarantees]]).

**Steps** (identifying $\mathbb R^2\cong\mathbb C$, spectral parameter $k\in\mathbb C$):

1. From the measured DtN map, compute the traces of the [[Complex Geometrical Optics Solutions]] $\psi(\cdot,k)$ on $\partial\Omega$ by solving a boundary integral equation.
2. Compute the non-physical **scattering transform**
   $$ \mathbf t(k) = \int_{\partial\Omega} e^{i\bar k\bar z}\,(\Lambda_\gamma-\Lambda_1)\,\psi(\cdot,k)\,\mathrm ds .$$
3. For each $z\in\Omega$, solve the $\bar\partial$-equation in $k$
   $$ \bar\partial_k\,\mu(z,k) = \frac{\mathbf t(k)}{4\pi\bar k}\,e_{-k}(z)\,\overline{\mu(z,k)}, \qquad e_k(z) = e^{i(kz+\bar k\bar z)},$$
   with $\mu(z,\cdot)\to1$ as $|k|\to\infty$.
4. Recover $\gamma(z) = \mu(z,0)^2$.

**Regularisation.** With noisy data, $\mathbf t(k)$ is only reliable for small $|k|$. Truncating to $|k|\le R$, with $R$ chosen according to the noise level, gives a provably convergent regularisation strategy (Knudsen et al. 2009). The result is a smoothed (low-pass) conductivity.

**Variants.** Approximations such as $\mathbf t^{\exp}$ replace step 1 with a Born-type approximation. *Deep D-bar* (Hamilton & Hauptmann 2018) post-processes the blurry D-bar image with a U-Net to sharpen edges (see [[Deep Learning for EIT]]).

## References

1. A. I. Nachman (1996). *Global Uniqueness for a Two-Dimensional Inverse Boundary Value Problem*. Ann. of Math. 143(1), 71–96. [doi:10.2307/2118653](https://doi.org/10.2307/2118653)
2. S. Siltanen, J. Mueller, D. Isaacson (2000). *An implementation of the reconstruction algorithm of A Nachman for the 2D inverse conductivity problem*. Inverse Problems 16(3), 681–699. [doi:10.1088/0266-5611/16/3/310](https://doi.org/10.1088/0266-5611/16/3/310)
3. K. Knudsen, M. Lassas, J. L. Mueller, S. Siltanen (2009). *Regularized D-bar method for the inverse conductivity problem*. Inverse Probl. Imaging 3(4), 599–624. [doi:10.3934/ipi.2009.3.599](https://doi.org/10.3934/ipi.2009.3.599)
4. J. L. Mueller, S. Siltanen (2012). *Linear and Nonlinear Inverse Problems with Practical Applications*. SIAM. [doi:10.1137/1.9781611972344](https://doi.org/10.1137/1.9781611972344)
5. S. J. Hamilton, A. Hauptmann (2018). *Deep D-Bar: Real-Time Electrical Impedance Tomography Imaging With Deep Neural Networks*. IEEE Trans. Med. Imaging 37(10), 2367–2377. [doi:10.1109/TMI.2018.2828303](https://doi.org/10.1109/TMI.2018.2828303)
