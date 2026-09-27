---
tags: [overview]
---

**Electrical Impedance Tomography (EIT)** recovers the conductivity $\sigma$ inside a
body $\Omega \subset \mathbb{R}^d$ from voltage measurements taken with electrodes on
its boundary $\partial\Omega$.

Currents are injected through the electrodes, the resulting voltages are measured, and
the pair (currents, voltages) is compared against the prediction of the
[[Forward Problem]].

## An ill-posed inverse problem

The map from $\sigma$ to boundary data, the *forward operator* $F$, is smooth but its
inverse is not continuous: small measurement noise can cause large reconstruction
errors. Stable reconstructions therefore minimise a regularised misfit

$$
\sigma^\ast = \arg\min_\sigma \; \tfrac12 \lVert F(\sigma) - U^\delta \rVert_2^2 + R(\sigma),
$$

where $R$ is e.g. [[Tikhonov Regularization]] or [[Total Variation]].

## Measurement setup

```tikz
\begin{document}
\begin{tikzpicture}
  \draw[thick] (0,0) circle (2);
  \fill[gray!30] (0.6,0.5) circle (0.6);
  \node at (0.6,0.5) {$\sigma_1$};
  \node at (-0.8,-0.6) {$\sigma_0$};
  \foreach \a in {0,45,...,315} {
    \fill (\a:2) circle (0.12);
  }
  \node[right] at (2.2,0) {$e_1$};
  \node[above right] at (45:2.2) {$e_2$};
  \draw[->, thick] (-3.2,0) -- (-2.2,0) node[midway, above] {$I$};
\end{tikzpicture}
\end{document}
```

## In the package

See `reconstruct` in the API reference, or the notebook-style *Getting Started* page.
