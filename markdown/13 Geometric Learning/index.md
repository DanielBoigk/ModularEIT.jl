---
title: Geometric Learning
tags: [overview, machine-learning, symmetry]
---

EIT on symmetric domains has symmetric data: rotating the conductivity rotates the measurements. Networks that respect these symmetries need fewer parameters and less data. And EIT domains are rarely rectangles: networks for EIT must cope with arbitrary shapes, meshes and uneven resolution.

**Reading order.**

1. The symmetries: [[Symmetries of the EIT Problem]], with the [[Dihedral Group D4]] of the square as example.
2. Invariance and equivariance: [[Invariant and Equivariant Functions]], [[Reynolds Operator]].
3. Architectures: [[Invariant Filter Banks]], [[Equivariant Convolutions]].
4. Networks for EIT domains: [[Networks on EIT Domains]], with [[Masked and Partial Convolutions]], [[Conformal Transplantation of Networks]] and [[Graph Convolutions on Finite Element Meshes]].

**Related.** Where these networks are used: [[11 Learned Priors/index|Learned Priors]]; rotational symmetry is also what makes the disk solver fast (see [[Fast Solvers on Disk Domains]]).

## References

1. M. M. Bronstein, J. Bruna, T. Cohen, P. Veličković (2021). *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges*. [arXiv:2104.13478](https://arxiv.org/abs/2104.13478)
