---
tags: [overview]
---

Notation used throughout the wiki.

| Symbol | Meaning |
|:--|:--|
| $\Omega\subset\mathbb R^n$ | bounded Lipschitz domain (body) |
| $\Gamma = \partial\Omega$, $\nu$ | boundary, outward unit normal |
| $\gamma$ | true conductivity |
| $\sigma$ | conductivity estimate / optimisation variable |
| $u$ | electric potential (voltage) in $\Omega$ |
| $\mathbf J = -\gamma\nabla u$ | current density |
| $f = u\vert_{\partial\Omega}$ | boundary voltage (Dirichlet data) |
| $g = \gamma\,\partial_\nu u\vert_{\partial\Omega}$ | boundary current density (Neumann data) |
| $\Lambda_\gamma$ | [[Dirichlet-to-Neumann Map]] $f\mapsto g$ |
| $\Lambda_\gamma^{-1} = \mathcal R_\gamma$ | [[Neumann-to-Dirichlet Map]] $g\mapsto f$ |
| $\varphi_i$ | finite element basis functions |
| $M_{ij} = \int\varphi_i\varphi_j$ | [[Mass Matrix]] |
| $K_{ij} = \int\nabla\varphi_i\cdot\nabla\varphi_j$ | [[Stiffness Matrix]] |
| $L_\gamma$, $(L_\gamma)_{ij} = \int\gamma\nabla\varphi_i\cdot\nabla\varphi_j$ | [[Weighted Stiffness Matrix]] |
| $M_\Gamma$, $K_\Gamma$ | [[Boundary Mass and Stiffness Matrices]] |
| $\lambda$ | adjoint state (Lagrange multiplier) |
| $J(\sigma)$ | data misfit functional |
| $\mathcal R(\sigma)$, $\beta$ | regulariser, regularisation parameter |
| $d(\cdot,\cdot)$ | data distance / metric |
| $J$ (matrix), $r$ | Jacobian and residual in [[Gauss-Newton Method|Gauss–Newton]] |
| $L_{\text{LM}}$ | [[Levenberg-Marquardt Method|Levenberg–Marquardt]] damping matrix |
| $\beta(t)$, $\bar\alpha(t)$ | diffusion [[Noise Schedule]] and cumulative signal factor |
| $\varepsilon_\theta(x,t)$ | noise-prediction network |
| $s_\theta(x,t)$ | learned [[Score Function]] |

Note the double use of $\beta$ (regularisation parameter vs. noise schedule) and of $J$ (misfit vs. Jacobian). The meaning is always clear from the context of the article.
