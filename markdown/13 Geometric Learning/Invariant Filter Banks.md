---
tags: [geometric-learning, architecture]
---

A convolution filter $w$ is **$D_4$-invariant** if $\rho(g)w = w$ for all $g\in D_4$, that is, if it is unchanged under $90°$ rotations and flips. Convolution with an invariant filter commutes with the group action on images, so $w*(\rho(g)x) = \rho(g)(w*x)$.

**Construction via the Reynolds operator.** Symmetrise any filter (see [[Reynolds Operator]]):

$$
w_{\text{inv}} = \frac18\sum_{g\in D_4}\rho(g)\,w .
$$

**Basis.** Applied to the single-pixel filters $\delta_{(i,j)}$, symmetrisation gives the normalised indicator of the $D_4$-orbit of $(i,j)$. These orbit indicators are orthogonal and form a **basis of the invariant filters**. The dimension equals the number of orbits.

**Example.** For a $17\times17$ filter restricted to a disc of radius 8 (offsets with $i^2+j^2\le64$):

- there are $197$ offsets;
- they split into $32$ orbits: $1$ of size 1 (the centre), $13$ of size 4 (on axes or diagonals), and $18$ of size 8. Check: $1+13\cdot4+18\cdot8 = 197$;
- so the invariant filter space is 32-dimensional, and any invariant $17\times17$ disc filter is a combination of 32 orthogonal basis filters.

**Limitation.** Invariant filters are isotropic up to $D_4$. They cannot detect *oriented* features such as edges of a particular direction. Stacking only invariant layers therefore loses directional information early. The remedy is to use [[Equivariant Convolutions]] first and apply invariance at the end (see [[Invariant and Equivariant Functions]]).

## References

1. T. Cohen, M. Welling (2016). *Group Equivariant Convolutional Networks*. ICML 2016. [arXiv:1602.07576](https://arxiv.org/abs/1602.07576)
2. M. Weiler, G. Cesa (2019). *General E(2)-Equivariant Steerable CNNs*. NeurIPS 32. [arXiv:1911.08251](https://arxiv.org/abs/1911.08251)
