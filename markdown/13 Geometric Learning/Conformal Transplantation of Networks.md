---
tags: [machine-learning, geometry, architecture, conformal]
aliases: [Shape-agnostic networks, Domain-agnostic U-Net, Conformal transplantation]
---

In two dimensions EIT on any simply connected domain $\Omega$ is equivalent to EIT on the unit disk $D$. With a conformal map $\Phi:D\to\Omega$, a conductivity $\sigma$ on $\Omega$ corresponds to the isotropic conductivity $\tilde\sigma = \sigma\circ\Phi$ on $D$, and the boundary data transform by transporting voltages and currents (see [[Conformal Invariance of the Conductivity Equation]]). A network designed for the disk, or for a square via a second map, therefore applies to *every* simply connected domain:

1. compute the map $\Phi$ for the domain (see [[Numerical Conformal Mapping]]);
2. pull back the conductivity (or the reconstruction to be denoised), $\tilde\sigma = \sigma\circ\Phi$, onto a fixed grid on $D$;
3. apply the network on $D$;
4. push the result forward, $\sigma = \tilde\sigma\circ\Phi^{-1}$.

This is the natural **shape-agnostic U-Net** for 2D EIT. The architecture never sees the shape, because the physics does not depend on it after the pull-back.

## What changes, and what does not

- **The equation does not change.** Unlike a general diffeomorphism, a conformal map keeps the conductivity isotropic and its values unchanged. A reconstruction method on $D$ remains a reconstruction method.
- **The resolution does.** A uniform grid on $D$ corresponds to a grid on $\Omega$ with local spacing $|\Phi'|$ times the reference spacing. Where the boundary bulges out, $|\Phi'|$ is large and the resolution is coarse. At re-entrant parts of a non-convex domain it is fine, and the conformal "crowding" of elongated domains can make it extremely uneven.
- **The image statistics do too.** A prior learned from images on $\Omega$ is not the same as one learned on $D$. Shapes are stretched by $|\Phi'|$ and rotated by $\arg\Phi'$. A prior trained on the reference domain therefore expects reference-domain statistics. It is exact for priors that are themselves conformally invariant, and approximately right when $|\Phi'|$ varies little.
- **Electrodes move.** The electrodes of $\Omega$ land at other positions on $\partial D$, with lengths scaled by $1/|\Phi'|$ and a varying contact impedance. The forward model on $D$ must use them.

For convex, roughly round domains (a thorax, a head cross-section) the distortion is mild and transplantation works well. For elongated or strongly non-convex domains a rectangle as reference domain, or networks on the mesh itself (see [[Graph Convolutions on Finite Element Meshes]]), avoid the crowding.

In three dimensions nothing comparable exists: by Liouville's theorem the only conformal maps are Möbius transformations.

## References

1. J. L. Mueller, S. Siltanen (2012). *Linear and Nonlinear Inverse Problems with Practical Applications*. SIAM. [doi:10.1137/1.9781611972344](https://doi.org/10.1137/1.9781611972344)
2. S. J. Hamilton, A. Hauptmann (2018). *Deep D-Bar: Real-Time Electrical Impedance Tomography Imaging With Deep Neural Networks*. IEEE Trans. Med. Imaging 37(10), 2367–2377. [doi:10.1109/TMI.2018.2828303](https://doi.org/10.1109/TMI.2018.2828303)
