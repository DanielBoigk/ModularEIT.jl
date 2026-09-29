---
tags: [analysis]
aliases: [H^1/2, Trace space]
---

The natural setting for the [[Conductivity Equation]] is the Sobolev space

$$
H^1(\Omega) = \{ u\in L^2(\Omega) : \nabla u \in L^2(\Omega)^n\},
\qquad \|u\|_{H^1}^2 = \|u\|_{L^2}^2 + \|\nabla u\|_{L^2}^2 .
$$

A function in $H^1(\Omega)$ has no pointwise boundary values. It does, however, have a well-defined **trace**. On a Lipschitz domain the trace operator

$$
\operatorname{tr}: H^1(\Omega) \to H^{1/2}(\partial\Omega), \qquad u\mapsto u|_{\partial\Omega},
$$

is bounded and surjective. Its image $H^{1/2}(\partial\Omega)$ is the space of admissible boundary *voltages*. Boundary *currents* $\gamma\,\partial_\nu u$ live in the dual space $H^{-1/2}(\partial\Omega)$. The pairing $\langle g, f\rangle$ extends $\int_{\partial\Omega} g f\,\mathrm ds$.

Useful subspaces:

- $H^1_0(\Omega)$: functions with zero trace (test space of the [[Dirichlet Problem]]);
- $H^1(\Omega)/\mathbb R$: functions modulo constants (solution space of the [[Neumann Problem]]);
- $H^{\pm 1/2}_\diamond(\partial\Omega)$: boundary functions with zero mean, the domain and range of the [[Neumann-to-Dirichlet Map]].

On a smooth closed curve, $H^{s}(\partial\Omega)$ can be described by Fourier coefficients: $\|f\|_{H^s}^2 \simeq \sum_k (1+k^2)^s |\hat f_k|^2$. This is the idea behind the [[Discrete Fractional Sobolev Norms]].

**Poincaré–Wirtinger inequality.** $\|u - \bar u\|_{L^2(\Omega)} \le C\|\nabla u\|_{L^2(\Omega)}$, where $\bar u$ is the mean of $u$. So $\|\nabla u\|_{L^2}$ is a norm on $H^1(\Omega)/\mathbb R$.

## References

1. L. C. Evans (2010). *Partial Differential Equations*, 2nd ed. AMS GSM 19. [doi:10.1090/gsm/019](https://doi.org/10.1090/gsm/019)
2. W. McLean (2000). *Strongly Elliptic Systems and Boundary Integral Equations*. Cambridge University Press. [ISBN 978-0-521-66375-5](https://search.worldcat.org/search?q=bn:9780521663755)
