---
tags: [forward-model]
---

The **forward problem** maps a conductivity $\sigma$ and a current pattern $I$ to the
electrode voltages $U = F(\sigma) I$. In ModularEIT it is modelled with the
[[Complete Electrode Model]].

## Weak formulation

Multiplying by test functions $(v, V) \in H^1(\Omega) \oplus \mathbb{R}^L$ and
integrating by parts gives the bilinear form

$$
B\big((u,U),(v,V)\big) = \int_\Omega \sigma \nabla u \cdot \nabla v \,\mathrm{d}x
+ \sum_{\ell=1}^L \frac{1}{z_\ell} \int_{e_\ell} (u - U_\ell)(v - V_\ell)\,\mathrm{d}s ,
$$

and the problem reads: find $(u, U)$ with $B\big((u,U),(v,V)\big) = \sum_\ell I_\ell V_\ell$
for all test functions.

## Sensitivity

Gradient-based reconstruction needs the Jacobian

$$
J_{\ell k} = \frac{\partial U_\ell}{\partial \sigma_k},
$$

which the package approximates by finite differences (`jacobian`). It is used by both
[[Tikhonov Regularization]] and [[Total Variation]] reconstructions.
