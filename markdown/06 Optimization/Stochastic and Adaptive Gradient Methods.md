---
tags: [optimization, machine-learning]
aliases: [Adam, RMSProp, SGD]
---

Methods from machine learning use only (possibly noisy) gradients, with per-coordinate adaptive step sizes:

- **SGD with momentum:** $m_{k+1} = \mu m_k+g_k$, $\sigma_{k+1} = \sigma_k-\eta m_{k+1}$.
- **RMSProp:** scales by a running root-mean-square of past gradients.
- **Adam:** exponential moving averages of the gradient ($m$) and its square ($v$), with bias correction:
  $$ m_{k+1} = \beta_1m_k+(1-\beta_1)g_k,\quad v_{k+1} = \beta_2v_k+(1-\beta_2)g_k^2,\quad \sigma_{k+1} = \sigma_k-\eta\frac{\hat m_{k+1}}{\sqrt{\hat v_{k+1}}+\epsilon}. $$
- **NAdam** adds Nesterov momentum to Adam; **AdaGrad** accumulates all past squared gradients.

**In EIT.** The objective is a sum over current patterns, $\Phi = \sum_iJ_i$. Using one pattern, or a random subset, per step gives an unbiased stochastic gradient at a fraction of the cost. This is the finite-sum setting of SGD. Stochastic objectives such as the [[RED-Diff]] regulariser also require these methods, since their gradients are random by construction.

**Caveats.** Adaptive methods do not use line searches or curvature information. On deterministic, ill-conditioned least-squares problems such as full-batch EIT, [[Gauss-Newton Method|Gauss–Newton]] or [[L-BFGS]] typically converge much faster. The coordinate-wise scaling of Adam also depends on the discretisation.

## References

1. D. P. Kingma, J. Ba (2015). *Adam: A Method for Stochastic Optimization*. ICLR 2015. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
2. L. Bottou, F. E. Curtis, J. Nocedal (2018). *Optimization Methods for Large-Scale Machine Learning*. SIAM Review 60(2), 223–311. [doi:10.1137/16M1080173](https://doi.org/10.1137/16M1080173)
